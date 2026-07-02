# Benchmarks — measured on the rig (4× DGX Spark GB10, TP=4, GMU 0.80)

## KV pool + max context

| KV dtype | max-model-len | GPU KV cache size | Max concurrency @ that len |
|----------|---------------|-------------------|----------------------------|
| fp8      | 262,144       | 965,376 tokens    | 3.68× |
| **nvfp4**| 262,144       | 1,317,376 tokens  | **5.03×** |
| **nvfp4**| **1,048,576** | 1,308,928 tokens  | **1.25×** (full 1M/request) |

nvfp4 grows the pool **+36%** over fp8 → a single request can hold the full 1M-token context.

## Decode throughput (thinking-off)

| Config | Single-stream | C4 aggregate |
|--------|---------------|--------------|
| fp8, 262K (base repo)            | ~33.7 tok/s | ~79 tok/s |
| nvfp4 Phase 1 (scratch-dequant)  | ~17.8 tok/s | ~39.8 tok/s |
| nvfp4 Phase 2 (inline dequant, eager) | ~25.2 tok/s | ~37.4 tok/s |
| nvfp4 Phase 2 + cudagraphs       | ~17.3 tok/s | ~22.7 tok/s (regression — see note) |

Phase 1 runs ~half of fp8 because of scratch-dequant (unpack KV to a bf16 scratch buffer every decode step).

Phase 2 = inline dequant: fuse the 4-bit→bf16 unpack into the attention kernel (no scratch pass). This recovered single-stream from 17.8 to 25.2 tok/s (+42%) while keeping the full 1M pool — bit-identical decode output to Phase 1. It is the shipped config (eager).

Cudagraph note: enabling nvfp4 cudagraphs forces `VLLM_USE_BREAKABLE_CUDAGRAPH`, which disables vLLM's torch.compile pipeline and net-regressed to 17.3 tok/s. So we serve eager (`--enforce-eager`). A full WMMA tensor-core decode path was also built but came out slower on M3's 1-KV-head-per-rank TP=4 layout (single-warp blocks) — shipped OFF.

## Coherence check (nvfp4 @ 1M)
Prompt: *"In one sentence: what makes a sneaker collectible?"*
> "A sneaker becomes collectible through a combination of limited availability, cultural significance, iconic design, celebrity or athlete association, and historical importance within sneaker culture."

Clean, on-topic, engine stable post-generation. ✅
