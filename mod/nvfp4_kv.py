# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 (4-bit) KV-cache helpers for MiniMax M3 on GB10/sm121.

EXPERIMENTAL — Phase 1 (correctness-first, scratch-dequant read path),
ported from the GB10-proven MiMo ``nvfp4-kv-diffkv`` mod.

Cache layout (both the M3 sparse backend and TRITON_ATTN full-attention
layers use the same logical 5-D shape):

    kv_cache: uint8 [num_blocks, 2, block_size, num_kv_heads, FULL_DIM]
              K = [:, 0], V = [:, 1]
    FULL_DIM = head_size // 2 + head_size // 16   (= 72 for head_size 128)

Per (side, token, head) row layout (72 bytes for head_size 128):

    [ fp4 data: head_size//2 bytes | fp8-e4m3 block scales: head_size//16 ]

i.e. the MiMo row-interleaved convention, NOT the region-separated
``[K_data | K_scale | V_data | V_scale]`` per-page layout used by
``csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`` /
``vllm.utils.torch_utils.nvfp4_kv_cache_split_views``.  The compiled CUDA
store kernel swizzles the V block scales for the SM100 trtllm-gen reader,
which nothing on sm121 can read back; this module's store and read are the
only producers/consumers of the M3 nvfp4 cache and are layout-consistent
with each other (flashinfer ``nvfp4_kv_quantize`` / ``nvfp4_kv_dequantize``
round-trip, global scale fixed at 1.0 — per-group-16 fp8 block scales carry
all magnitude, same as the MiMo recipe, K/V recon rel-err ~0.095).

Capture-safety:
  * ``nvfp4_store_kv`` / ``bf16_scatter_rows``  — static shapes, in-kernel
    slot<0 skip, no host branches -> cudagraph-capturable (same contract as
    MiMo's ``_nvfp4_store_diffkv``).
  * ``nvfp4_dequant_blocks`` / ``nvfp4_sparse_scratch`` — data-dependent
    (torch.unique + data-sized scratch alloc): NOT capturable. They must run
    either inside an ``eager_break_during_capture`` region (the M3 sparse
    layer's ``_run_attention`` and the generic ``unified_attention_with_output``
    are both decorated) or with attention cudagraphs disabled. The patched
    metadata builders therefore report ``AttentionCGSupport.NEVER`` for nvfp4
    unless ``VLLM_M3_NVFP4_ALLOW_CG=1``.
"""

import os

import torch

from vllm.triton_utils import tl, triton

# Env knob: "1" keeps each builder's native cudagraph-support level under
# nvfp4 (only safe with VLLM_USE_BREAKABLE_CUDAGRAPH=1, where the attend
# paths run as eager break points). Default "0": report NEVER -> attention
# runs eager / piecewise-only.
NVFP4_ALLOW_CG = os.environ.get("VLLM_M3_NVFP4_ALLOW_CG", "0") == "1"


def nvfp4_full_dim(head_size: int) -> int:
    """Packed uint8 last dim: fp4 data + fp8 block scales."""
    return head_size // 2 + head_size // 16


_NVFP4_KV_GS: dict = {}


def _nvfp4_gs(device: torch.device) -> torch.Tensor:
    """Global scale tensor, fixed at 1.0 (block scales carry magnitude)."""
    g = _NVFP4_KV_GS.get(device)
    if g is None:
        g = torch.tensor(1.0, device=device, dtype=torch.float32)
        _NVFP4_KV_GS[device] = g
    return g


# ---------------------------------------------------------------------------
# Store path (capturable)
# ---------------------------------------------------------------------------
@triton.jit
def _nvfp4_scatter_kernel(packed_ptr, slot_ptr, cache_ptr, NH, BS, SB,
                          s_pt, s_ph, s_cb, s_cs, s_ch,
                          BLK: tl.constexpr):
    # grid (T*NH,): write packed[t, h, :SB] -> cache[slot//BS, slot%BS, h, :SB]
    # In-kernel slot<0 skip (no host sync) + static shapes => capturable.
    pid = tl.program_id(0)
    t = pid // NH
    h = pid % NH
    slot = tl.load(slot_ptr + t)
    if slot < 0:
        return
    blk = slot // BS
    off = slot % BS
    o = tl.arange(0, BLK)
    m = o < SB
    src = tl.load(packed_ptr + t * s_pt + h * s_ph + o, mask=m, other=0)
    dst = cache_ptr + blk * s_cb + off * s_cs + h * s_ch + o
    tl.store(dst, src, mask=m)


def nvfp4_store_kv(
    key: torch.Tensor,        # [T, NH, HD] bf16 (may be a strided view)
    value: torch.Tensor,      # [T, NH, HD] bf16
    kv_cache: torch.Tensor,   # uint8 [nb, 2, BS, NH, HD//2 + HD//16]
    slot_mapping: torch.Tensor,  # [T] int64 (slot<0 = skip)
) -> None:
    """Quantize K/V to nvfp4 and scatter into the packed paged cache."""
    from flashinfer import nvfp4_kv_quantize

    T, NH, HD = key.shape
    BS = kv_cache.shape[2]
    fb, sb = HD // 2, HD // 16
    SB = fb + sb
    gs = _nvfp4_gs(key.device)
    kf, ks = nvfp4_kv_quantize(key.reshape(-1, HD).contiguous(), gs)
    vf, vs = nvfp4_kv_quantize(value.reshape(-1, HD).contiguous(), gs)
    if ks.dtype != torch.uint8:
        ks = ks.view(torch.uint8)
        vs = vs.view(torch.uint8)
    kp = torch.cat([kf.view(T, NH, fb), ks.view(T, NH, sb)], dim=2).contiguous()
    vp = torch.cat([vf.view(T, NH, fb), vs.view(T, NH, sb)], dim=2).contiguous()
    BLK = triton.next_power_of_2(SB)
    for side, packed in ((0, kp), (1, vp)):
        c = kv_cache[:, side]  # [nb, BS, NH, SB] strided view (K or V plane)
        _nvfp4_scatter_kernel[(T * NH,)](
            packed, slot_mapping, c, NH, BS, SB,
            packed.stride(0), packed.stride(1),
            c.stride(0), c.stride(1), c.stride(2),
            BLK=BLK,
        )


@triton.jit
def _row_scatter_kernel(src_ptr, slot_ptr, cache_ptr, D,
                        s_st, BLK: tl.constexpr):
    # grid (T,): write src[t, :D] -> cache_rows[slot, :D] (rows contiguous).
    t = tl.program_id(0)
    slot = tl.load(slot_ptr + t)
    if slot < 0:
        return
    o = tl.arange(0, BLK)
    m = o < D
    x = tl.load(src_ptr + t * s_st + o, mask=m, other=0)
    tl.store(cache_ptr + slot * D + o, x, mask=m)


def bf16_scatter_rows(
    x: torch.Tensor,           # [T, D] bf16 (may be a strided view of qkv)
    cache: torch.Tensor,       # bf16, flat rows of D per slot (e.g. [nb, BS, D])
    slot_mapping: torch.Tensor,  # [T] int64 (slot<0 = skip)
) -> None:
    """Scatter bf16 rows by slot (indexer side-cache insert).

    The M3 indexer cache is addressed flat (``index_cache + slot * head_dim``)
    by the fused CUDA kernel, so rows are contiguous per slot.
    """
    T, D = x.shape
    assert cache.stride(-1) == 1
    BLK = triton.next_power_of_2(D)
    _row_scatter_kernel[(T,)](
        x, slot_mapping, cache, D, x.stride(0), BLK=BLK,
    )


# ---------------------------------------------------------------------------
# Read path (scratch dequant; eager/non-captured only)
# ---------------------------------------------------------------------------
def nvfp4_dequant_blocks(
    kv_cache: torch.Tensor,   # uint8 [nb, 2, BS, NH, SB]
    active: torch.Tensor,     # [nact] int64, unique physical block ids
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequant the given physical blocks into a bf16 scratch cache.

    Returns ``[nact, 2, BS, NH, HD]`` (same logical layout the bf16 attend
    kernels expect), where ``HD = (SB * 8 // 9) * 2``.
    """
    from flashinfer import nvfp4_kv_dequantize

    _, _, BS, NH, SB = kv_cache.shape
    fb = SB * 8 // 9          # fp4 data bytes  (64 for head_size 128)
    sb = SB - fb              # fp8 scale bytes (8 for head_size 128)
    HD = fb * 2
    gs = _nvfp4_gs(kv_cache.device)
    nact = int(active.numel())
    if nact == 0:
        return torch.zeros(
            1, 2, BS, NH, HD, dtype=out_dtype, device=kv_cache.device
        )
    blk = kv_cache[active]    # [nact, 2, BS, NH, SB] contiguous gather
    data = blk[..., :fb].reshape(-1, fb).contiguous()
    scale = blk[..., fb:].reshape(-1, sb).contiguous()
    # flashinfer's nvfp4_kv_dequantize only supports bf16/fp16 output
    # (DISPATCH_DLPACK_DTYPE). Anything else (fp32, or an fp8-quantized query
    # dtype that leaks in) must be dequanted to bf16 then cast. Callers pass
    # the query dtype; the query-quant disable in triton_attn keeps it bf16 on
    # the real path, so the cast is normally a no-op.
    if out_dtype in (torch.bfloat16, torch.float16):
        deq_dtype = out_dtype
    else:
        deq_dtype = torch.bfloat16
    deq = nvfp4_kv_dequantize(data, scale, gs, output_dtype=deq_dtype)
    if deq_dtype != out_dtype:
        deq = deq.to(out_dtype)
    return deq.reshape(nact, 2, BS, NH, HD)


def _remap_block_table(
    bt: torch.Tensor, remap: torch.Tensor, num_blocks: int
) -> torch.Tensor:
    """Map physical block ids -> scratch indices (padding-safe via clamp)."""
    # NOTE: no in-place ops -- ``bt`` may alias the live block table.
    return remap[bt.to(torch.int64).clamp(0, num_blocks - 1)]


def _topk_physical_pages(
    topk_idx: torch.Tensor,   # [H, TQ, K] logical block ids, -1 = padding
    block_table: torch.Tensor,  # [num_reqs, MB]
    req_of_token: torch.Tensor,  # [TQ] int64
) -> torch.Tensor:
    """Physical page ids referenced by the top-k selection (1-D, valid only)."""
    bt_rows = block_table.index_select(0, req_of_token).to(torch.int64)  # [TQ, MB]
    lt = topk_idx.clamp(min=0).to(torch.int64)  # [H, TQ, K]
    mb = bt_rows.shape[1]
    lt = lt.clamp_(max=mb - 1)  # safety: never index past the table row
    phys = torch.gather(bt_rows.unsqueeze(0).expand(lt.shape[0], -1, -1), 2, lt)
    return phys[topk_idx >= 0].reshape(-1)


def nvfp4_sparse_scratch(
    kv_cache: torch.Tensor,
    main_md,                      # MiniMaxM3SparseMetadata
    decode_topk: torch.Tensor | None,
    prefill_topk: torch.Tensor | None,
    out_dtype: torch.dtype,
):
    """Scratch-dequant exactly the topk-referenced blocks for the M3 sparse
    attend and remap the decode/prefill block tables onto the scratch.

    Returns ``(scratch_kv, decode_block_table, prefill_block_table)`` — drop-in
    replacements for the kv_cache / block tables consumed by
    ``minimax_m3_sparse_attn(_decode)``.

    Data-dependent (unique + data-sized alloc): must run outside cudagraph
    capture. The M3 sparse layer calls this inside its
    ``@eager_break_during_capture`` ``_run_attention``.
    """
    num_blocks = kv_cache.shape[0]
    dev = kv_cache.device
    pages = []
    d_bt = None
    p_bt = None

    if main_md.num_decodes > 0 and decode_topk is not None:
        d = main_md.decode
        d_bt = d.block_table
        tq = decode_topk.shape[1]
        req = (
            torch.arange(tq, device=dev, dtype=torch.int64)
            // d.decode_query_len
        )
        pages.append(_topk_physical_pages(decode_topk, d_bt, req))

    if main_md.num_prefills > 0 and prefill_topk is not None:
        p = main_md.prefill
        p_bt = p.block_table
        tq = prefill_topk.shape[1]
        cu = p.cu_seqlens_q.to(torch.int64).contiguous()
        pos = torch.arange(tq, device=dev, dtype=torch.int64)
        # token i belongs to request j iff cu[j] <= i < cu[j+1]
        req = torch.searchsorted(cu[1:], pos, right=True)
        pages.append(_topk_physical_pages(prefill_topk, p_bt, req))

    if not pages:
        scratch = nvfp4_dequant_blocks(
            kv_cache, torch.empty(0, dtype=torch.int64, device=dev), out_dtype
        )
        return scratch, d_bt, p_bt

    active = torch.unique(torch.cat(pages).clamp_(0, num_blocks - 1))
    scratch = nvfp4_dequant_blocks(kv_cache, active, out_dtype)
    remap = torch.zeros(num_blocks, dtype=torch.int32, device=dev)
    remap[active] = torch.arange(
        int(active.numel()), dtype=torch.int32, device=dev
    )
    if d_bt is not None:
        d_bt = _remap_block_table(d_bt, remap, num_blocks)
    if p_bt is not None:
        p_bt = _remap_block_table(p_bt, remap, num_blocks)
    return scratch, d_bt, p_bt


def nvfp4_full_attn_scratch(
    kv_cache: torch.Tensor,     # uint8 [nb, 2, BS, NH, SB]
    block_table: torch.Tensor,  # [num_reqs, MB]
    out_dtype: torch.dtype,
):
    """Scratch-dequant every block referenced by the batch's block table
    (full attention reads the whole context) + remapped block table.

    Returns ``(scratch_kv, remapped_block_table)``.
    """
    num_blocks = kv_cache.shape[0]
    # NOTE: no in-place ops -- ``block_table`` aliases the live block table.
    bt64 = block_table.to(torch.int64).clamp(0, num_blocks - 1)
    active = torch.unique(bt64)
    scratch = nvfp4_dequant_blocks(kv_cache, active, out_dtype)
    remap = torch.zeros(
        num_blocks, dtype=block_table.dtype, device=block_table.device
    )
    remap[active] = torch.arange(
        int(active.numel()), dtype=block_table.dtype, device=block_table.device
    )
    return scratch, remap[bt64]
