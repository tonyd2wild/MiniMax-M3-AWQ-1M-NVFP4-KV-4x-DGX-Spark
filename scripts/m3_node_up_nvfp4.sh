#!/bin/bash
# Start the M3 TP=4 container on this node and join/start the Ray cluster.
# Usage: m3_node_up.sh <head|worker> <self_fabric_ip> <head_fabric_ip> <nccl_gid_index>
set -euo pipefail
ROLE="$1"; SELF_IP="$2"; HEAD_IP="$3"; GID="$4"
IMAGE=minimax-m3-awq:tp4-sm121-nvfp4kv
NAME=m3-tp4
HFCACHE="$HOME/.cache/huggingface"

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --restart no --network host --ipc host --shm-size 64gb --gpus all \
  --device /dev/infiniband:/dev/infiniband \
  --ulimit memlock=-1 \
  --cap-add IPC_LOCK \
  -v "$HFCACHE":/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  -e VLLM_HOST_IP="$SELF_IP" \
  -e NCCL_IB_HCA=rocep1s0f0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_GID_INDEX="$GID" \
  -e NCCL_CROSS_NIC=1 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=WARN \
  "$IMAGE" sleep infinity

sleep 4
# Patch transformers ALLOWED_LAYER_TYPES so M3's custom 'minimax_m3_sparse' layer type
# passes the strict validator in transformers 5.9 (else config load fails).
CFG=/opt/env/lib/python3.12/site-packages/transformers/configuration_utils.py
docker exec "$NAME" grep -q '"minimax_m3_sparse"' "$CFG" \
  || docker exec "$NAME" sed -i '62a\    "minimax_m3_sparse",' "$CFG"
echo "layer_types patch count: $(docker exec "$NAME" grep -c minimax_m3_sparse "$CFG")"

echo "container $NAME started on $SELF_IP; IB + GPU check:"
docker exec "$NAME" ls /dev/infiniband/ >/dev/null 2>&1 && echo "  /dev/infiniband OK" || echo "  /dev/infiniband MISSING"
docker exec "$NAME" python3 -c "import torch; print('  cuda', torch.cuda.is_available(), 'n', torch.cuda.device_count())"

if [ "$ROLE" = head ]; then
  docker exec "$NAME" ray start --head --node-ip-address "$SELF_IP" --port 6379 \
    --num-gpus 1 --disable-usage-stats
else
  docker exec "$NAME" ray start --address "$HEAD_IP:6379" --node-ip-address "$SELF_IP" \
    --num-gpus 1 --disable-usage-stats
fi
echo "=== NODE $SELF_IP ($ROLE) UP ==="
