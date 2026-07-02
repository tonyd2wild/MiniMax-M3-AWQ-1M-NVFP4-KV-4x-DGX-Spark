# fix-minimax-m3-nvfp4-kv — Phase 2: INLINE dequant read path

Eliminates the Phase-1 scratch-dequant pass: the packed nvfp4 KV is now
dequantized **inside the attention kernels** (registers, fused with the
attention math). Store path + packed layout are IDENTICAL to Phase 1
(`[fp4 64B | fp8-e4m3 scales 8B]` per (side, token, head) row, MiMo
convention, global scale 1.0) — only the READ side changed, so a cache
written by either build is readable by both.

Base: the Phase-1 mod (`m3-nvfp4-mod/fix-minimax-m3-nvfp4-kv/`, see its
NOTES.md for layout/plumbing) on vLLM commit `4c626633` (`~/m3-build/vllm`
on Asusi). Reference for the inline technique: the GB10-proven MiMo
`nvfp4-kv-diffkv` mod (`triton_unified_attention_diffkv.py` IS_NVFP4 path +
`wmma_decode.py`).

## What changed (kernel level)

**The inline read (all three kernel families).** At each K/V tile load the
kernels now read the PACKED uint8 rows directly and dequantize in-flight:

1. load the fp4 data bytes (`[D/2, TILE]` for K, `[TILE, D/2]` for V);
2. widen to int32, shift-and-mask the two nibbles per byte
   (`low nibble = dim 2b`, `high = 2b+1` — verified EXACTLY against
   flashinfer's packing, see Validation), reshape to `[D, TILE]`/`[TILE, D]`;
3. decode e2m1 via a 16-entry float32 LUT gather (`nvfp4_e2m1_lut`);
4. load the fp8-e4m3 block scales **once per 16-dim group**
   (`[D/16, TILE]`, via an fp8-typed view of the same cache buffer) and
   broadcast-expand in registers;
5. `K = (lut[nib] * scale).to(q.dtype)` feeds the existing `tl.dot`s.

No bf16 scratch tensor, no `torch.unique`, no gather, no block-table remap,
no data-sized allocations — per decode step the packed bytes are read ONCE
by the attention kernel itself (previously: flashinfer unpack pass writing
a bf16 scratch [2.28x the packed bytes] + kernel re-reading that scratch,
plus ~6 host-side tensor ops per layer per step, x60 layers).

| file | change |
|---|---|
| `sparse_attn.py` (**new to mod**, patches `vllm/models/minimax_m3/common/ops/sparse_attn.py`) | `_gqa_sparse_fwd_kernel` + `_gqa_sparse_decode_kernel`: `USE_NVFP4` constexpr branch at the K/V loads (steps 1-5 above); wrappers detect the packed cache by shape (uint8 & last dim != head_dim), pass the fp8 scale view + LUT, and pin `num_stages=1` for nvfp4 (the extra staged loads on top of the 128x128 dot tiles exceed the 100KB shared-memory limit with multi-stage pipelining — first build hit `OutOfResources: 303112 > 101376`, fixed by group-scale loads + 1 stage). |
| `triton_unified_attention.py` (**new to mod**, patches `vllm/v1/attention/ops/`) | `kernel_unified_attention`: `IS_NVFP4` constexpr branch (same 5 steps) in the non-TD tile load, works in both 2D and 3D (split-KV decode) launches; `KV_QUANT_MODE` stays NONE so all existing quant predicates are untouched. Wrapper: `nvfp4_packed=` kwarg builds the fp8 views + LUT, clamps TILE_SIZE to 16 (32 measured slower for the unpack path — same finding as MiMo), and hosts the optional WMMA hook. |
| `wmma_decode_m3.py` (**new**, installs to `vllm/v1/attention/ops/`) | MiMo's WMMA tensor-core flash-decode ported to M3: `Hk=Hv=128, G=16, SB=72`, template `BS in {16,32,64,128}`, TWO-plane addressing (K/V are separate strided planes of one buffer; kernel takes both base pointers + explicit block stride). Capture-safe (fixed NSPLIT, static scratch, no JIT compile during capture). **Default OFF** — see Bench. |
| `sparse_attention.py` | nvfp4 forward branch: inline (default) passes the PACKED cache + original block tables straight to the kernels; `VLLM_M3_NVFP4_INLINE=0` restores the Phase-1 scratch path verbatim. Builder `get_cudagraph_support`: native support restored for nvfp4 under inline (reads are now static-shape); NEVER only for the scratch fallback. |
| `triton_attn.py` | same split for the 3 full layers (+ EAGLE3 drafter): inline passes `kv_cache.unbind(1)` packed planes + `nvfp4_packed=True`; scratch fallback unchanged. Builder CG likewise restored under inline. `supports_quant_query_input` stays False for nvfp4 (query must remain bf16). |
| `nvfp4_kv.py` | added `NVFP4_INLINE` env knob + `nvfp4_e2m1_lut` (cached float32[16]). Store helpers + scratch helpers unchanged (scratch kept as fallback). |
| `model.py`, `indexer.py` | unchanged from Phase 1 (store path only). |

## Why this recovers speed

Phase 1's ~2x decode loss (33.7 -> 17.8 tok/s) was NOT attention math — it
was the per-step read overhead: every decode step, every layer ran
`torch.unique` + gather + flashinfer unpack into a fresh bf16 scratch +
block-table remap (6+ eager host ops x 60 layers), and attention cudagraphs
were forced off. Inline dequant removes all of it:

- **57 sparse layers**: offline, one decode step (4 seqs x 16K ctx, topk 8)
  went **0.528ms -> 0.045ms (11.8x)** vs the scratch path.
- **3 full layers**: single-seq decode kernel time, scratch -> inline:
  1.5x @16K, 1.66x @64K, **1.85x @262K** (the scratch's extra
  write+read of a 2.28x-sized bf16 buffer is gone; the packed cache is
  now also the *only* thing read: 144B/token vs fp8's 256B/token, so at
  long ctx this path is bandwidth-*better* than fp8).
- **Cudagraphs**: reads are static-shape now, so the builders report their
  native support again (UNIFORM_BATCH / ALWAYS) — the eager-attention tax
  disappears wherever the deployment captures decode.
- Store path unchanged (was already capture-safe and fast).

## Offline validation (scratch container `kai-nvfp4-scratch` on Asusi, image `minimax-m3-awq:tp4-sm121`, GB10/sm121 — live `m3-tp4` untouched)

1. **Packing convention** (`tests/test_convention.py`): torch emulation of
   the in-kernel unpack (nibble order + LUT + fp8 scale) vs flashinfer
   `nvfp4_kv_dequantize` on flashinfer-quantized data:
   `max|diff| = 0` (bit-exact). Recon rel-err vs original 0.0952 = the
   known quant noise (matches Phase-1/MiMo ~0.095).
2. **Full-attn kernels** (`tests/test_unified.py`), inline vs Phase-1
   scratch reference:
   - decode 3D split-KV path, bs 64/128/16: **rel = 0.0 (bit-identical)**
   - decode 2D path: rel 8.0e-4; MTP q_len=3: rel 5.7e-4; prefill
     (q_len 128, mixed context): rel 6.9e-4 — all pure bf16
     accumulation-order noise from the tile-size difference (16 vs 32).
3. **Sparse kernels** (`tests/test_sparse.py`), inline vs scratch:
   decode q_len 1 & 2: **rel = 0.0 (bit-identical)**; prefill
   (2 seqs, prefix contexts): rel 2.2e-5.
4. **WMMA** (`tests/test_wmma.py`): compiles on sm121 in-image, correct on
   bs 16/64/128 + MTP q_len 3 (rel ~1.6e-3 vs Triton = quant-noise level,
   same as MiMo's 0.0026).
5. flashinfer symbols present in-image (run.sh asserts).

## Bench summary (offline, single GB10 sharing memory with the live model)

| shape | Phase-1 scratch | Triton inline | WMMA |
|---|---|---|---|
| sparse decode step, 4x16K, topk8 | 0.528ms | **0.045ms (11.8x)** | n/a |
| full-attn decode, 1x16K | 0.178ms | **0.118ms (1.5x)** | 0.428ms |
| full-attn decode, 1x64K | 0.733ms | **0.441ms (1.7x)** | 1.283ms |
| full-attn decode, 1x262K | 4.19ms | **2.27ms (1.9x)** | 4.44ms |

**WMMA verdict for M3**: correct but SLOWER than Triton inline at every
context tested (NSPLIT 64/128/512 all tried). Root cause: M3 TP=4 has
NKVH=1 per rank, so the kernel's grid is (1, NSPLIT, num_seqs) blocks of a
single warp — too little work per SM vs MiMo's NKVH=2/Hk=192 where it won
2.3x. Shipped default-OFF (`VLLM_M3_WMMA_DECODE=1` to experiment); the
fallback plumbing is battle-tested (MiMo ran it for weeks).

## How Kai applies it

Identical to Phase 1 but with THIS dir: build image from
`minimax-m3-awq:tp4-sm121` running `bash <mod>/run.sh`
(SITE_PACKAGES=/opt/env/lib/python3.12/site-packages), serve with
`--kv-cache-dtype nvfp4`. Knobs:

- `VLLM_M3_NVFP4_INLINE=0` -> exact Phase-1 behavior (scratch read,
  CG NEVER). The A/B safety valve.
- `VLLM_M3_WMMA_DECODE=1` -> experimental WMMA decode for the 3 full layers
  (not recommended per bench above).
- `VLLM_M3_NVFP4_ALLOW_CG` now only matters for the scratch fallback.

Window-test checklist:
- startup: same KV pool line as Phase 1 (1.32M / 5.03x @262K expected —
  memory accounting unchanged);
- coherence at short + long ctx (needle test — quantization is identical
  to Phase 1, so quality should match Phase 1 exactly; the 3D decode and
  sparse decode paths are bit-identical to it);
- **tok/s**: expect most of the fp8-baseline ~33.7 tok/s back at short ctx;
  at very long ctx inline nvfp4 reads fewer bytes than fp8 (144 vs 256
  B/token) so the full-attn read is cheaper than fp8's, though the fp8
  deployment may still win elsewhere (MSA vs Triton sparse impl);
- MTP/EAGLE3 spec decode explicitly (q_len>1 verify batches hit the 2D
  inline path);
- if the deployment runs FULL cudagraph mode, watch startup capture — the
  builders now report native CG support under inline. If capture
  misbehaves, set `VLLM_M3_NVFP4_INLINE=0` (full Phase-1 revert, eager
  scratch reads) and report.

## Risks / unknowns for the live window test

1. **Cudagraph re-enable is the biggest behavioral delta** vs Phase 1 (which
   ran attention eagerly). The kernels themselves are static-shape and the
   sparse decode's partial-buffer allocs mirror the stock bf16 path (which
   captures fine), but nvfp4 + capture hasn't run end-to-end. Mitigation:
   serve once with `VLLM_M3_NVFP4_INLINE=1` + `--enforce-eager` (or
   cudagraph off) to isolate, then enable CG.
2. Triton recompile storm on first requests (new constexpr variants:
   USE_NVFP4 x tile sizes). One-time; warms up with the first
   prefill+decode.
3. The 2D/prefill inline path differs from scratch by bf16 rounding only
   (~7e-4) — no quality impact expected (quantization error is 100x
   larger), but the long-ctx needle test is still the arbiter.
4. `num_stages=1` on the sparse kernels applies ONLY to the nvfp4 variant
   (constexpr-keyed cache); bf16/fp8 variants keep default stages.
5. WMMA off by default = zero new CUDA code on the hot path; the .cu only
   compiles if someone sets VLLM_M3_WMMA_DECODE=1.

## Test artifacts

`tests/` in this dir: `test_convention.py`, `test_unified.py`,
`test_sparse.py`, `test_wmma.py`, `bench_longctx.py` — all runnable inside
a scratch container after `run.sh` (they import the installed modules).
Host copy synced at `tonyspark3@Asusi:~/kai-nvfp4-inline/`.
