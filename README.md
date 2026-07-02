# MiniMax-M3-AWQ at 1M-token context on 4× DGX Spark (GB10) — via a 4-bit nvfp4 KV cache

Serving **MiniMax-M3-AWQ-INT4 at a full `1,048,576`-token context window** across **4× NVIDIA DGX Spark (GB10, `sm_121a`)**, TP=4 — unlocked by swapping the KV cache from fp8 to **4-bit `nvfp4`**.

**This is live and serving** at `http://100.90.25.78:8000/v1` (model `minimax-m3-awq`, `--max-model-len 1048576`).

> ## Credit
> This is a KV-cache extension on top of **[CosmicRaisins' MiniMax-M3-AWQ recipe](https://github.com/CosmicRaisins/minimax-m3-awq-gb10)** — the model, mods, and serve flags are theirs. See our base TP=4 build here: **[MiniMax-M3-AWQ-TP4-EAGLE3-4x-DGX-Spark](https://github.com/tonyd2wild/MiniMax-M3-AWQ-TP4-EAGLE3-4x-DGX-Spark)**. Concurrency-tuning lineage from **Keys ([@u1tra_instinct](https://x.com/u1tra_instinct))**. The nvfp4-KV packing convention is ported from the MiMo nvfp4-KV work.
>
> **What THIS repo adds** = the **nvfp4 4-bit KV cache** for M3 that grows the KV pool ~36% and lets a single request run the full **1M-token** context on the same 4 Sparks.

---

## The headline

| | fp8 KV (base repo) | **nvfp4 KV (this repo)** |
|---|---|---|
| KV pool @ GMU 0.80 | 965,376 tokens | **1,308,928 tokens (+36%)** |
| Max context / request | 262,144 | **1,048,576 (1M)** |
| Concurrency @ that context | 3.68× @ 262K | **1.25× @ 1M** |
| Decode tok/s (single) | ~33.7 | ~25.2 *(Phase 2 inline; Phase 1 was ~17.8)* |
| Generation | coherent | coherent ✅ |

**nvfp4 KV nearly halves per-token KV memory → the pool grows enough that one request can hold a full 1M-token context.** Same 4× GB10 hardware, same model, coherent output.

## How it works

- **Packed nvfp4 KV**: main KV of all attention layers (57 sparse + 3 full) + the EAGLE3 drafter is stored 4-bit `nvfp4`, row-interleaved `[fp4 64B | fp8-scales 8B]` (MiMo-lineage layout — deliberately **not** the SM100-swizzled `reshape_and_cache_nvfp4` path, which nothing on sm121 can read back). The lightning-indexer side-cache stays bf16 (quantizing it too is Phase 2 → the full ~2×).
- **The load-bearing fix**: vLLM's generic `Attention` layer auto-quantizes the **query** to fp8 whenever the KV dtype is fp8/nvfp4. That fp8 query was being fed into flashinfer's `nvfp4_kv_dequantize` (which only dispatches bf16/fp16) → `DISPATCH_DLPACK_DTYPE` crash on the first generation. Setting `supports_quant_query_input = False` for nvfp4 keeps the query **bf16**, and the full-attention path becomes identical to stock bf16 attention on the dequanted scratch. **That one line is what made it generate cleanly.**
- **Serve**: `--kv-cache-dtype nvfp4 --max-model-len 1048576` (+ pin `VLLM_ATTENTION_BACKEND=TRITON_ATTN`). Everything else = the base M3 recipe (cu130-rebuilt vLLM, TP=4 Ray, Marlin, EAGLE3, GMU 0.80).

## Honest tradeoff (Phase 1 vs Phase 2)

This is a **Phase-1, correctness-first** implementation: it uses **scratch-dequant** — every decode step, the referenced nvfp4 KV blocks are unpacked into a bf16 scratch buffer, then the existing attention kernels run on that. That extra unpack pass costs speed: **~17.8 tok/s single-stream vs fp8's ~33.7** (about half).

**Phase 2 = inline / WMMA dequant** — fuse the unpack *into* the attention kernel (dequantize each block on-the-fly in registers/shared memory, no scratch pass) to recover most of the tok/s while keeping the 1M pool. In progress.

So today this is a **long-context / big-pool** build (1M tokens at ~18 tok/s), not a speed build. Phase 2 aims to make it both.

## Repo layout

```
README.md            — this file
RECIPE.md            — build the nvfp4 image (base sm121 image + the mod) → serve at 1M
BENCHMARKS.md        — measured pool + tok/s (fp8 vs nvfp4, 262K vs 1M)
mod/                 — the nvfp4-KV mod (run.sh installer + patched vLLM files + NOTES.md)
scripts/
  m3_node_up_nvfp4.sh    — per-node container + Ray bring-up (nvfp4 image)
  m3_serve_nvfp4_1m.sh   — head-node vllm serve, --kv-cache-dtype nvfp4 --max-model-len 1048576
```

_2Wild / Topia Labs. Built on CosmicRaisins' M3 recipe; nvfp4-KV layout from the MiMo nvfp4-KV work; concurrency lineage Keys (@u1tra_instinct)._
