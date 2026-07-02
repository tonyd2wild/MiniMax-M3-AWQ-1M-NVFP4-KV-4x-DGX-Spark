# fix-minimax-m3-nvfp4-kv — Phase 1 port notes

## ROUND 2 FIX (read first) — dequant crash on generation

**Round-1 result:** loads with `--kv-cache-dtype nvfp4`, KV pool grew **+34%**
(965,376 -> 1,291,904 tokens, 4.93x @ 262K) — memory win + plumbing proven.
BUT generation crashed the engine in `nvfp4_dequant_blocks ->
flashinfer.nvfp4_kv_dequantize -> DISPATCH_DLPACK_DTYPE`.

**Root cause (found, not guessed):** the generic `Attention` layer
(`vllm/model_executor/layers/attention/attention.py`) builds a `QuantFP8`
`query_quant` whenever `kv_cache_dtype in {"fp8","fp8_e4m3","nvfp4"}` **and**
`impl.supports_quant_query_input` is True (True on CUDA). So for the **3 full
attention layers** the query is quantized to **fp8** before `impl.forward`,
making `query.dtype == float8_e4m3fn`. My full-attn read passed that dtype as
the dequant `output_dtype`; flashinfer's `nvfp4_kv_dequantize` only dispatches
bf16/fp16 (verified in-image: `output_dtype=float32` or fp8 both raise the
exact `DISPATCH_DLPACK_DTYPE`/"failed to dispatch data type" error;
bf16 & fp16 pass). The `\x02` in the coordinator's trace is the fp8 query
tensor's DLPack type code, not fp32. Even with the dtype coerced, an fp8 query
vs my **bf16** dequanted scratch (`kv_quant_mode=NONE`) would be a matmul
dtype mismatch. The **sparse** layers never hit this — they don't use the
generic `Attention` layer; their query is `qkv.new_empty(...)` (bf16) passed
straight to the impl.

**Fix (2 changes):**
1. `triton_attn.py` `TritonAttentionImpl.__init__`: for
   `self._kv_quant_mode.is_nvfp4`, set `self.supports_quant_query_input =
   False`. This makes the generic layer skip building `query_quant`, so the
   query stays **bf16**, `output_dtype` stays bf16, and my bf16 dequanted
   scratch + `kv_quant_mode=NONE` + `q_descale=None` path is fully consistent
   with the vanilla bf16 attention path (which is exactly what we want: we
   dequant KV to bf16 and run plain bf16 attention). The `k_descale/v_descale`
   I pass are `layer._k_scale.expand(...)` = 1.0, identical to the stock bf16
   branch.
2. `nvfp4_kv.py` `nvfp4_dequant_blocks`: defensive — dequant to bf16 (or fp16
   if requested) and `.to(out_dtype)` only if needed. Belt-and-suspenders so
   an unexpected dtype degrades gracefully instead of crashing the engine.
   With fix #1 the query is bf16 on the real path, so this cast is a no-op.

**Validated in the scratch container (`docker exec m3-tp4`):**
- `nvfp4_kv_dequantize` signature `(fp4_data u8[M,K/2], block_scales u8[M,K/16],
  global_scale f32, output_dtype=bf16) -> [M,K]`; matches my calls exactly.
- output_dtype bf16 OK, fp16 OK, **fp32 FAILS** with the reported error;
  quantize input bf16/fp16 OK, fp32 FAILS. Model `torch_dtype: bfloat16`.
- **End-to-end round-trip** (my exact packing: `cat([fp4|scales])` store +
  gather/slice `[:fb]`/`[fb:]` read): K recon rel-err **0.0895**, matching the
  direct flashinfer round-trip (~0.09) -> the packed layout stores and reads
  back coherently.

No change to the store side (it already worked: key/value are bf16 at insert,
`query_quant` only touched the query).

---

# Phase 1 port notes (round 1)

nvfp4 (4-bit) KV cache for **MiniMax-M3-AWQ on 4x DGX Spark (GB10/sm121)**,
ported from the GB10-proven MiMo `nvfp4-kv-diffkv` mod
(`~/.openclaw/workspace/mimo-tp2-500k-nvfp4kv-spark/recipe/mods/nvfp4-kv-diffkv/`).

Phase 1 scope: packed nvfp4 on the **main KV of both attention paths**
(57 sparse layers + 3 full TRITON_ATTN layers + any EAGLE3 drafter full-attn
layers). The lightning-indexer side cache **stays bf16**. Read path =
**scratch dequant** (MiMo's correct-first path), NOT inline/WMMA (Phase 2).

Base tree: vLLM commit `4c626633` (`~/m3-build/vllm` on Asusi; installed at
`/opt/env/lib/python3.12/site-packages/vllm/` in the image). All patched files
below were diffed against that tree.

## Files in this mod

| file | installs to (under site-packages) | what changed |
|---|---|---|
| `nvfp4_kv.py` | `vllm/models/minimax_m3/common/ops/nvfp4_kv.py` | **NEW.** All nvfp4 helpers: `nvfp4_store_kv` (flashinfer `nvfp4_kv_quantize` + Triton scatter, slot<0 skip, capturable), `bf16_scatter_rows` (indexer key insert), `nvfp4_dequant_blocks` (flashinfer `nvfp4_kv_dequantize` of gathered blocks -> bf16 scratch), `nvfp4_sparse_scratch` (topk-referenced blocks only + remapped decode/prefill block tables), `nvfp4_full_attn_scratch` (all batch-referenced blocks + remapped table), `NVFP4_ALLOW_CG` env knob. |
| `sparse_attention.py` | `vllm/models/minimax_m3/common/sparse_attention.py` | `MiniMaxM3SparseBackend`: `"nvfp4"` added to `supported_kv_cache_dtypes`; `get_kv_cache_shape` returns packed `(nb, 2, bs, nkv, 72)` for nvfp4 (72 = `nvfp4_kv_cache_full_dim(128)`). Builder: `get_cudagraph_support` returns `NEVER` for nvfp4 (unless `VLLM_M3_NVFP4_ALLOW_CG=1`). `MiniMaxM3SparseImpl.__init__`: `use_fp8_kv` no longer true for nvfp4 (it would have viewed the packed cache as fp8). `MiniMaxM3SparseTritonImpl.forward`: nvfp4 branch dequants only topk-referenced blocks -> bf16 scratch + remapped block tables, then calls the **unchanged** `minimax_m3_sparse_attn(_decode)` kernels. `select_main_impl_cls`: MSA excluded for nvfp4. |
| `indexer.py` | `vllm/models/minimax_m3/common/indexer.py` | `"nvfp4"` added to `MiniMaxM3IndexerBackend.supported_kv_cache_dtypes` (validation-permissive only; side-cache storage stays bf16 — `indexer_kv_dtype` is a separate knob and remains `"bf16"`). |
| `model.py` | `vllm/models/minimax_m3/nvidia/model.py` | `MiniMaxM3SparseAttention`: `use_nvfp4_kv` flag; `forward` nvfp4 branch calls the fused `qknorm_rope_kv_insert` op **without insert** (`kv_cache=None`; the kernel still norms+ropes k and index_k in place in `qkv` — verified in the .cu: only V skips the in-place store, and V needs no transform), then `nvfp4_store_kv(k, v, main cache)` + `bf16_scatter_rows(index_k, indexer cache)`. bf16/fp8 path untouched. The MTP drafter (`nvidia/mtp.py`, `force_sparse_attn=True`) reuses this class, so it is covered automatically. |
| `triton_attn.py` | `vllm/v1/attention/backends/triton_attn.py` | `TritonAttentionBackend`: `"nvfp4"` in supported list (the list at the old line 254); packed `get_kv_cache_shape`. Builder `get_cudagraph_support` -> `NEVER` for nvfp4 (env-overridable). `TritonAttentionImpl.forward`: nvfp4 branch dequants all batch-referenced blocks -> bf16 scratch + remapped block table, passes `kv_quant_mode=NONE` into the unchanged `unified_attention`. `do_kv_cache_update`: nvfp4 -> `nvfp4_store_kv`. `fused_rope_kvcache_supported` -> False for nvfp4. Covers the 3 full layers **and** an EAGLE3 drafter (standard `Attention` layers select TRITON_ATTN). |
| `run.sh` | — | installer: backs up originals as `*.orig-nvfp4kv`, copies the 5 files, ast-checks each, verifies flashinfer has `nvfp4_kv_quantize`/`nvfp4_kv_dequantize`. **No monkeypatch needed**: unlike the 0.21-era MiMo mod, this tree's generic `Attention.get_kv_cache_spec` already resolves nvfp4 -> uint8 storage + `KVQuantMode.NVFP4` (`kv_cache_dtype_str_to_dtype` / `get_kv_quant_mode`), and `AttentionSpec.real_page_size_bytes` already budgets the packed page size. |

## Cache layout (deliberate choice — read this before Phase 2)

Per (side, token, head) row of 72 bytes: `[fp4 data 64B | fp8-e4m3 block
scales 8B]` — the **MiMo row-interleaved convention**, self-consistent between
this mod's store and read (flashinfer quantize/dequantize round-trip, global
scale fixed 1.0; checkpoint k_scale/v_scale are ignored, same as MiMo —
validated there at K/V recon rel-err ~0.095).

We did **NOT** use the tree's compiled CUDA store kernel
(`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`, reachable via
`reshape_and_cache_flash(..., "nvfp4")`) nor the
`torch_utils.nvfp4_kv_cache_split_views` region-separated page layout
(`[K_data | K_scale | V_data | V_scale]`): that kernel **swizzles the V block
scales** for the SM100 trtllm-gen reader, which nothing on sm121 can read
back. Since this mod's store+read are the only producers/consumers of the M3
nvfp4 cache, the simpler proven layout wins for Phase 1. Phase 2 could switch
to the CUDA store kernel + a swizzle-aware read for speed.

Allocation sizing is handled by the existing spec plumbing
(`real_page_size_bytes` for `KVQuantMode.NVFP4` = `2*bs*nkv*72` — exactly
`prod(get_kv_cache_shape[1:])`), so store/read layout is invisible to the
KV-cache manager.

## Expected pool gain

Per token per rank (TP=4 -> 1 KV head/rank, head_dim 128, vs the current
**fp8** deployment):

- sparse layer: main 2x128B -> 2x72B, indexer +256B bf16 (unchanged): 512 -> 400 B
- full layer: 512 -> 144 B

57 sparse + 3 full: `(57*512 + 3*512) / (57*400 + 3*144)` = 30720/23232 ≈
**1.32x** KV pool (≈ **1.29x** if the indexer rows are budgeted separately,
which they are — the indexer group is untouched). Check the startup
`"GPU KV cache size"` log line grows accordingly. (vs a bf16 baseline it
would be ~2x.)

## How Kai applies it

1. Build a new image from `minimax-m3-awq:tp4-sm121` that runs
   `bash fix-minimax-m3-nvfp4-kv/run.sh` (default
   `SITE_PACKAGES=/opt/env/lib/python3.12/site-packages`; override if the
   layout differs). Tag `minimax-m3-awq:tp4-sm121-nvfp4kv`. Do NOT touch the
   live m3-tp4 container.
2. Serve identically to the live config but with `--kv-cache-dtype nvfp4`.
3. Smoke sequence:
   - startup completes; log shows the sparse backend selected Triton
     (`MiniMax M3 sparse attention selected Triton (kv_cache_dtype=nvfp4 ...)`)
     and `"GPU KV cache size"` grew ~1.3x vs the fp8 run at the same
     `gpu_memory_utilization`;
   - single short generation is coherent (garble here = kernel/layout bug —
     stop and report);
   - a >128-token prompt (multiple KV blocks, exercises prefill scratch) and
     a multi-request batch;
   - long-context window test + bench (expect decode slowdown at long ctx —
     see Known costs).

## Known costs / behavior changes (Phase 1, by design)

- **Scratch dequant on every read.** Sparse decode touches only ~topk blocks
  per seq (cheap). The 3 full layers dequant the whole referenced context
  per step (~100KB/blk-token... concretely ≈ `nblocks * 128KB` bf16 scratch
  traffic per layer per step at 1 KV head/rank): expect a decode tok/s hit
  that grows with context. Prefill dequants effectively the whole context per
  sparse layer per chunk too. Phase 2 (inline fp4 dequant in-kernel, MiMo's
  `triton_unified_attention_diffkv.py` / `wmma_decode.py` show how) removes
  this.
- **Attention cudagraphs disabled for nvfp4** (builders report `NEVER`):
  the scratch read is data-dependent (`torch.unique`, data-sized allocs).
  The M3 sparse attend already runs inside `@eager_break_during_capture`
  (`_run_attention`), and `unified_attention_with_output` is likewise an
  eager break point, so **if** the deployment uses
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, set `VLLM_M3_NVFP4_ALLOW_CG=1` to keep
  full-CG decode (the store path was kept capture-safe: static shapes,
  in-kernel slot<0 skip, flashinfer quantize is capturable per the MiMo mod).
  Do NOT set ALLOW_CG=1 under the standard (non-breakable) FULL cudagraph
  mode — capture would crash (loudly, at startup, not silently).

## Open questions / risks for the window test

1. **flashinfer API drift (highest risk).** I could not import the image's
   flashinfer (no container access). The mod assumes the MiMo-recipe
   signatures: `nvfp4_kv_quantize(x_2d, global_scale) -> (u8[N,D/2],
   scales[N,D/16])` and `nvfp4_kv_dequantize(data, scales, global_scale,
   output_dtype=...)`, with uint8-viewable scales. `run.sh` asserts the
   symbols exist; if the signatures moved, fix up `nvfp4_kv.py` only.
2. **Backend selection for the full layers.** The tree's FLASHINFER backend
   also lists nvfp4 but hard-requires sm100f (raises "requires sm100f" at
   impl init). If startup crashes there, pin
   `VLLM_ATTENTION_BACKEND=TRITON_ATTN`.
3. **Spec-decode verify batches** (decode_query_len > 1): the topk gather
   maps token->request via `idx // decode_query_len` (request-major flatten,
   matches the kernel's own mapping). Believed correct, but exercise MTP
   spec-decode explicitly in the window test.
4. **Block-table padding**: assumed padded entries are >= 0 (vLLM pads with
   0); padded/clamped ids merely dequant an extra block. If a table ever
   contains negative ids the clamps keep it safe.
5. **Quantizing V as well as K on ALL layers** is more aggressive than fp8;
   watch long-context quality (needle tests) in the window test, not just
   coherence.
6. **kv-scale weights**: checkpoint k/v scales are bypassed (global scale
   1.0, per-16 fp8 block scales carry magnitude) — same convention MiMo ran
   for weeks.
7. `get_kv_cache_stride_order` HND layout: store/read use tensor strides
   everywhere, so both NHD/HND should work, but only NHD (the default) is
   what MiMo proved on GB10.

## Phase 2 candidates (not in this mod)

- Inline fp4 dequant inside `_gqa_sparse_fwd_kernel`/`_gqa_sparse_decode_kernel`
  and `unified_attention` (port MiMo's inline path + e2m1 LUT), killing the
  scratch.
- Use the compiled `reshape_and_cache_nvfp4` CUDA store (+ swizzle-aware read).
- Quantize the indexer side cache (`indexer_kv_dtype`, needs the CuteDSL
  indexer impl per `select_indexer_impl_cls`).
- Re-enable full cudagraph support once reads are capture-safe.
