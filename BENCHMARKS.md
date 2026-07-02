# Benchmarks — measured on the rig (4× DGX Spark GB10, TP=4, GMU 0.80)

## KV pool + max context

| KV dtype | max-model-len | GPU KV cache size | Max concurrency @ that len |
|----------|---------------|-------------------|----------------------------|
| fp8      | 262,144       | 965,376 tokens    | 3.68× |
| **nvfp4**| 262,144       | 1,317,376 tokens  | **5.03×** |
| **nvfp4**| **1,048,576** | 1,308,928 tokens  | **1.25×** (full 1M/request) |

nvfp4 grows the pool **+36%** over fp8 → a single request can hold the full 1M-token context.

## Decode throughput (thinking-off, single node measure)

| Config | Single-stream | C4 aggregate |
|--------|---------------|--------------|
| fp8, 262K            | ~33.7 tok/s | ~79 tok/s |
| **nvfp4, 262K** (Phase 1) | ~17.8 tok/s | ~39.8 tok/s |

**Phase-1 nvfp4 runs ~half of fp8** because of scratch-dequant (unpack KV→bf16 every decode step). **Phase 2 (inline/WMMA dequant)** targets recovering this while keeping the +36% pool / 1M context.

## Coherence check (nvfp4 @ 1M)
Prompt: *"In one sentence: what makes a sneaker collectible?"*
> "A sneaker becomes collectible through a combination of limited availability, cultural significance, iconic design, celebrity or athlete association, and historical importance within sneaker culture."

Clean, on-topic, engine stable post-generation. ✅
