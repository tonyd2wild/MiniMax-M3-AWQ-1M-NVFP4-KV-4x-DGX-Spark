# RECIPE — nvfp4 KV cache for MiniMax-M3-AWQ at 1M context

Prereq: the base build from **[MiniMax-M3-AWQ-TP4-EAGLE3-4x-DGX-Spark](https://github.com/tonyd2wild/MiniMax-M3-AWQ-TP4-EAGLE3-4x-DGX-Spark)** — you need the `minimax-m3-awq:tp4-sm121` image + the model/drafter staged in the HF cache on all 4 nodes. This recipe layers the nvfp4 KV cache on top.

## 1. Build the nvfp4 image (on each of the 4 nodes)

The mod is pure Python/Triton — no recompile. Apply it into the base image and commit:

```bash
# per node:
docker rm -f m3-nvfp4build 2>/dev/null
docker run -d --name m3-nvfp4build minimax-m3-awq:tp4-sm121 sleep infinity
docker cp mod m3-nvfp4build:/opt/m3-nvfp4mod
docker exec m3-nvfp4build bash -c \
  'export PATH=/opt/env/bin:$PATH SITE_PACKAGES=/opt/env/lib/python3.12/site-packages; cd /opt/m3-nvfp4mod && bash run.sh'
docker commit m3-nvfp4build minimax-m3-awq:tp4-sm121-nvfp4kv
docker rm -f m3-nvfp4build
```

`run.sh` installs the 5 patched files + asserts the image's flashinfer ships `nvfp4_kv_quantize`/`nvfp4_kv_dequantize` (it does, in the base sm121 image). See `mod/NOTES.md` for the file-by-file change list and the round-2 fix rationale.

## 2. Bring up the cluster (nvfp4 image) — all 4 nodes

```bash
# args: <head|worker> <self_fabric_ip> <head_fabric_ip> <nccl_gid_index>
# Asusi (head):
./scripts/m3_node_up_nvfp4.sh head   192.168.192.3 192.168.192.3 3
# Bluey / Spark4 (GID 3):
./scripts/m3_node_up_nvfp4.sh worker 192.168.192.1 192.168.192.3 3
./scripts/m3_node_up_nvfp4.sh worker 192.168.192.4 192.168.192.3 3
# Reddie (GID 5):
./scripts/m3_node_up_nvfp4.sh worker 192.168.192.2 192.168.192.3 5
```

`m3_node_up_nvfp4.sh` = the base `m3_node_up.sh` with `IMAGE=minimax-m3-awq:tp4-sm121-nvfp4kv` + `-e VLLM_ATTENTION_BACKEND=TRITON_ATTN` (keep the full-attn layers off FlashInfer's sm100-gated nvfp4 path).

## 3. Serve at 1M context (head node)

```bash
./scripts/m3_serve_nvfp4_1m.sh
# = the base serve command with:  --kv-cache-dtype nvfp4  --max-model-len 1048576
# tail ~/m3-serve.log for "GPU KV cache size: 1,308,928 tokens" +
#                         "Maximum concurrency for 1,048,576 tokens per request: 1.25x"
```

For 262K (higher concurrency, 3.68×→ effectively ~4.9× with nvfp4) instead of 1M, set `--max-model-len 262144`.

## 4. Verify

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"minimax-m3-awq","messages":[{"role":"user","content":"one sentence: what makes a sneaker collectible?"}],"max_tokens":80}'
```

## Recovery / notes
- After any engine crash, **full re-cycle** (re-run `m3_node_up_nvfp4.sh` on all 4, then serve) — re-serving a crashed Ray cluster fails.
- GMU **0.80** (0.85 OOMs at runtime on GB10). Load ~5-6 min (241 GB across TP=4).
- To roll back to fp8: re-cycle with the base `m3_node_up.sh`/`m3_serve.sh` (image `:tp4-sm121`, `--kv-cache-dtype fp8`).
