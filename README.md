# MiniMax-M3-AWQ on 4× DGX Spark — 1M-token context via a 4-bit nvfp4 KV cache

> Serving **MiniMax-M3-AWQ-INT4 at a full `1,048,576`-token context window** across **4× NVIDIA DGX Spark (GB10, `sm_121a`)**, TP=4 — unlocked by swapping the KV cache from fp8 to **4-bit `nvfp4`**.

**This is live and serving** at `http://100.90.25.78:8000/v1` (model `minimax-m3-awq`, `--max-model-len 1048576`).

> ### Credit
> This is a KV-cache extension on top of **[CosmicRaisins' MiniMax-M3-AWQ recipe](https://github.com/CosmicRaisins/minimax-m3-awq-gb10)** — the model, mods, and serve flags are theirs. See our base TP=4 build here: **[MiniMax-M3-AWQ-TP4-EAGLE3-4x-DGX-Spark](https://github.com/tonyd2wild/MiniMax-M3-AWQ-TP4-EAGLE3-4x-DGX-Spark)**. Concurrency-tuning lineage from **Keys ([@u1tra_instinct](https://x.com/u1tra_instinct))**. The nvfp4-KV packing convention is ported from the MiMo nvfp4-KV work.
>
> **What THIS repo adds** = the **nvfp4 4-bit KV cache** for M3 that grows the KV pool ~36% and lets a single request run the full **1M-token** context on the same 4 Sparks.

## TL;DR

- **What you get:** MiniMax-M3-AWQ-INT4 served at a full **1,048,576 (1M)**-token context on **4× DGX Spark (GB10, `sm_121a`)**, TP=4 — same hardware and model as the fp8 base build, coherent output.
- **Why it works:** the **nvfp4 4-bit KV cache** nearly halves per-token KV memory, growing the pool **+36%** (`965,376 → 1,308,928 tokens` @ GMU 0.80) — enough for one request to hold the entire 1M-token window.
- **Speed:** ~**25.2 tok/s** single-stream (Phase 2 inline dequant; Phase 1 scratch-dequant was ~17.8) vs fp8's ~33.7. This is a **long-context / big-pool** build, not a speed build.
- **Who it's for:** anyone who needs a single 1M-token request (or ~5× concurrency at 262K) on a 4× GB10 rig.

## Hardware

- **4× NVIDIA DGX Spark (GB10, `sm_121a`)**, TP=4, `--distributed-executor-backend ray`.
- **Fabric:** InfiniBand (`/dev/infiniband`, `NCCL_NET=IB`) on the `192.168.192.0/24` fabric; per-node `NCCL_IB_HCA=rocep1s0f0`, `NCCL_SOCKET_IFNAME=enp1s0f0np0`, GID index 3 (Asusi/Bluey/Spark4) or 5 (Reddie).
- **Memory:** GMU **0.80** (0.85 OOMs at runtime on GB10). Weights ~**241 GB** loaded across TP=4; load takes ~5-6 min.

## Quick start

Prereq: the base build from **[MiniMax-M3-AWQ-TP4-EAGLE3-4x-DGX-Spark](https://github.com/tonyd2wild/MiniMax-M3-AWQ-TP4-EAGLE3-4x-DGX-Spark)** — the `minimax-m3-awq:tp4-sm121` image + the model/drafter staged in the HF cache on all 4 nodes. This recipe layers the nvfp4 KV cache on top.

```bash
# 1. Build the nvfp4 image on each of the 4 nodes (pure Python/Triton — no recompile):
docker rm -f m3-nvfp4build 2>/dev/null
docker run -d --name m3-nvfp4build minimax-m3-awq:tp4-sm121 sleep infinity
docker cp mod m3-nvfp4build:/opt/m3-nvfp4mod
docker exec m3-nvfp4build bash -c \
  'export PATH=/opt/env/bin:$PATH SITE_PACKAGES=/opt/env/lib/python3.12/site-packages; cd /opt/m3-nvfp4mod && bash run.sh'
docker commit m3-nvfp4build minimax-m3-awq:tp4-sm121-nvfp4kv
docker rm -f m3-nvfp4build

# 2. Bring up the cluster — all 4 nodes:
#    args: <head|worker> <self_fabric_ip> <head_fabric_ip> <nccl_gid_index>
./scripts/m3_node_up_nvfp4.sh head   192.168.192.3 192.168.192.3 3   # Asusi (head)
./scripts/m3_node_up_nvfp4.sh worker 192.168.192.1 192.168.192.3 3   # Bluey  (GID 3)
./scripts/m3_node_up_nvfp4.sh worker 192.168.192.4 192.168.192.3 3   # Spark4 (GID 3)
./scripts/m3_node_up_nvfp4.sh worker 192.168.192.2 192.168.192.3 5   # Reddie (GID 5)

# 3. Serve at 1M context (head node):
./scripts/m3_serve_nvfp4_1m.sh
#    tail ~/m3-serve.log for "GPU KV cache size: 1,308,928 tokens" +
#                            "Maximum concurrency for 1,048,576 tokens per request: 1.25x"

# 4. Smoke test:
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"minimax-m3-awq","messages":[{"role":"user","content":"one sentence: what makes a sneaker collectible?"}],"max_tokens":80}'
```

Full build/serve walkthrough: **[RECIPE.md](RECIPE.md)**.

## Setup (detailed)

### Weights

- Model: `cyankiwi/MiniMax-M3-AWQ-INT4` (served as `minimax-m3-awq`).
- EAGLE3 drafter: `Inferact/MiniMax-M3-EAGLE3` (`num_speculative_tokens: 2`).
- Both are staged in the HF cache on all 4 nodes by the base build (see prereq above).

### Image / build (the mod)

The mod is pure Python/Triton — apply it into the base `minimax-m3-awq:tp4-sm121` image and commit as `minimax-m3-awq:tp4-sm121-nvfp4kv` (Quick start step 1). `run.sh` installs the patched vLLM files and asserts the image's flashinfer ships `nvfp4_kv_quantize`/`nvfp4_kv_dequantize` (it does, in the base sm121 image). See **[mod/NOTES.md](mod/NOTES.md)** for the file-by-file change list.

What the mod does:

- **Packed nvfp4 KV:** the main KV of all attention layers (57 sparse + 3 full) + the EAGLE3 drafter is stored 4-bit `nvfp4`, row-interleaved `[fp4 64B | fp8-scales 8B]` (MiMo-lineage layout — deliberately **not** the SM100-swizzled `reshape_and_cache_nvfp4` path, which nothing on sm121 can read back). The lightning-indexer side-cache stays bf16 (quantizing it too is Phase 2 → the full ~2×).
- **Phase 2 (shipped): inline dequant** — the packed nvfp4 KV is dequantized **inside the attention kernels** (registers, fused with the attention math), removing Phase 1's per-step scratch-dequant pass. Store path and packed layout are identical to Phase 1, so a cache written by either build is readable by both. The **[mod-inline-phase2/](mod-inline-phase2/)** dir is the shipped read path; `VLLM_M3_NVFP4_INLINE=0` restores the Phase-1 scratch read verbatim.

### Launch

`scripts/m3_serve_nvfp4_1m.sh` runs (on the head node, in a `tmux` session logging to `~/m3-serve.log`):

```bash
docker exec m3-tp4 vllm serve cyankiwi/MiniMax-M3-AWQ-INT4 \
  --served-model-name minimax-m3-awq \
  --trust-remote-code \
  --block-size 128 \
  --attention-backend TRITON_ATTN \
  --kv-cache-dtype nvfp4 \
  --language-model-only \
  --host 0.0.0.0 --port 8000 \
  -tp 4 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --enable-prefix-caching \
  --enforce-eager \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  --speculative-config '{"method":"eagle3","model":"Inferact/MiniMax-M3-EAGLE3","num_speculative_tokens":2,"attention_backend":"TRITON_ATTN"}'
```

Everything else = the base M3 recipe (cu130-rebuilt vLLM, TP=4 Ray, Marlin, EAGLE3, GMU 0.80). The node containers pin `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (keeps the full-attn layers off FlashInfer's sm100-gated nvfp4 path); `m3_node_up_nvfp4.sh` = the base `m3_node_up.sh` with `IMAGE=minimax-m3-awq:tp4-sm121-nvfp4kv` + that env.

### Verify

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"minimax-m3-awq","messages":[{"role":"user","content":"one sentence: what makes a sneaker collectible?"}],"max_tokens":80}'
```

Coherence check (nvfp4 @ 1M) — prompt *"In one sentence: what makes a sneaker collectible?"*:

> "A sneaker becomes collectible through a combination of limited availability, cultural significance, iconic design, celebrity or athlete association, and historical importance within sneaker culture."

Clean, on-topic, engine stable post-generation. ✅

## Benchmarks

Measured on the rig (4× DGX Spark GB10, TP=4, GMU 0.80).

### KV pool + max context

| KV dtype | max-model-len | GPU KV cache size | Max concurrency @ that len |
|----------|---------------|-------------------|----------------------------|
| fp8      | 262,144       | 965,376 tokens    | 3.68× |
| **nvfp4**| 262,144       | 1,317,376 tokens  | **5.03×** |
| **nvfp4**| **1,048,576** | 1,308,928 tokens  | **1.25×** (full 1M/request) |

nvfp4 grows the pool **+36%** over fp8 → a single request can hold the full 1M-token context.

### Decode throughput (thinking-off)

| Config | Single-stream | C4 aggregate |
|--------|---------------|--------------|
| fp8, 262K (base repo)                 | ~33.7 tok/s | ~79 tok/s |
| nvfp4 Phase 1 (scratch-dequant)       | ~17.8 tok/s | ~39.8 tok/s |
| nvfp4 Phase 2 (inline dequant, eager) | ~25.2 tok/s | ~37.4 tok/s |
| nvfp4 Phase 2 + cudagraphs            | ~17.3 tok/s | ~22.7 tok/s (regression — see Troubleshooting) |

Phase 1 runs ~half of fp8 because of scratch-dequant (unpack KV to a bf16 scratch buffer every decode step). Phase 2 fuses the 4-bit→bf16 unpack into the attention kernel (no scratch pass): +42% single-stream (17.8 → 25.2 tok/s) while keeping the full 1M pool, bit-identical decode output to Phase 1. It is the shipped config (eager). Full numbers in **[BENCHMARKS.md](BENCHMARKS.md)**.

## Configuration

Knobs and tradeoffs (serve flags in `m3_serve_nvfp4_1m.sh`, env in `m3_node_up_nvfp4.sh`):

- **`--max-model-len 1048576`** = full 1M per request (concurrency 1.25×). For higher concurrency instead of 1M, set **`--max-model-len 262144`** (3.68× → effectively ~4.9× with nvfp4).
- **`--kv-cache-dtype nvfp4`** — the 4-bit KV cache (this repo). Roll back with `fp8` on the base image.
- **`--gpu-memory-utilization 0.80`** — 0.85 OOMs at runtime on GB10.
- **`--max-num-seqs 4`**, **`--max-num-batched-tokens 8192`**, **`--block-size 128`**.
- **`--enforce-eager`** — served eager; enabling nvfp4 cudagraphs net-regresses (see Troubleshooting).
- **`VLLM_ATTENTION_BACKEND=TRITON_ATTN`** (pinned in the node containers) — keeps the full-attn layers off FlashInfer's sm100-gated nvfp4 path.
- **`VLLM_M3_NVFP4_INLINE`** (Phase 2, default `1` = inline dequant; `0` = Phase-1 scratch read — the A/B safety valve).
- **`VLLM_M3_WMMA_DECODE`** (default `0`; `1` = experimental WMMA tensor-core decode for the 3 full layers — measured **slower** than Triton inline on M3's TP=4 / 1-KV-head-per-rank shapes, shipped OFF).
- **`VLLM_M3_NVFP4_ALLOW_CG`** — only matters under breakable cudagraphs / the scratch fallback.
- **Spec decode:** EAGLE3, `num_speculative_tokens: 2`, `attention_backend: TRITON_ATTN`.

## Troubleshooting

- **Dequant crash on first generation** (`DISPATCH_DLPACK_DTYPE`). vLLM's generic `Attention` layer auto-quantizes the **query** to fp8 whenever the KV dtype is fp8/nvfp4; that fp8 query fed into flashinfer's `nvfp4_kv_dequantize` (which only dispatches bf16/fp16) crashed on the first generation. **Fix:** set `supports_quant_query_input = False` for nvfp4 in `triton_attn.py`, keeping the query **bf16** so the full-attention path is identical to stock bf16 attention on the dequanted scratch. **That one line is what made it generate cleanly.** (Full root-cause in [mod/NOTES.md](mod/NOTES.md).)
- **FlashInfer backend raises "requires sm100f"** at impl init for the full layers → pin `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (already set in `m3_node_up_nvfp4.sh`).
- **Config load fails on the custom layer type** — the node script patches transformers' `ALLOWED_LAYER_TYPES` to accept M3's `minimax_m3_sparse` type (transformers 5.9's strict validator), else config load fails.
- **Cudagraph regression.** Enabling nvfp4 cudagraphs forces `VLLM_USE_BREAKABLE_CUDAGRAPH`, which disables vLLM's torch.compile pipeline and net-regressed to 17.3 tok/s. Serve eager (`--enforce-eager`).
- **Triton recompile storm on first requests** (new nvfp4 constexpr variants × tile sizes) — one-time; warms up with the first prefill + decode.
- **After any engine crash, full re-cycle** — re-run `m3_node_up_nvfp4.sh` on all 4 nodes, then serve. Re-serving a crashed Ray cluster fails.
- **Roll back to fp8** — re-cycle with the base `m3_node_up.sh` / `m3_serve.sh` (image `:tp4-sm121`, `--kv-cache-dtype fp8`).

## Repo layout

```
README.md            — this file
RECIPE.md            — build the nvfp4 image (base sm121 image + the mod) → serve at 1M
BENCHMARKS.md        — measured pool + tok/s (fp8 vs nvfp4, 262K vs 1M)
mod/                 — Phase-1 nvfp4-KV mod (scratch-dequant): run.sh installer + patched vLLM files + NOTES.md
mod-inline-phase2/   — Phase-2 nvfp4-KV mod (inline dequant, shipped): run.sh + patched files + NOTES.md
scripts/
  m3_node_up_nvfp4.sh    — per-node container + Ray bring-up (nvfp4 image)
  m3_serve_nvfp4_1m.sh   — head-node vllm serve, --kv-cache-dtype nvfp4 --max-model-len 1048576
```

## Credits & links

- **[CosmicRaisins' MiniMax-M3-AWQ recipe](https://github.com/CosmicRaisins/minimax-m3-awq-gb10)** — the model, mods, and serve flags are theirs; this repo is a KV-cache extension on top.
- **[MiniMax-M3-AWQ-TP4-EAGLE3-4x-DGX-Spark](https://github.com/tonyd2wild/MiniMax-M3-AWQ-TP4-EAGLE3-4x-DGX-Spark)** — our base TP=4 build (prerequisite).
- **Keys ([@u1tra_instinct](https://x.com/u1tra_instinct))** — concurrency-tuning lineage.
- **MiMo nvfp4-KV work** — the nvfp4-KV packing convention and `nvfp4-kv-diffkv` inline-dequant technique are ported from it.
- Model: `cyankiwi/MiniMax-M3-AWQ-INT4`. EAGLE3 drafter: `Inferact/MiniMax-M3-EAGLE3`.

_2Wild / Topia Labs. Built on CosmicRaisins' M3 recipe; nvfp4-KV layout from the MiMo nvfp4-KV work; concurrency lineage Keys (@u1tra_instinct)._
