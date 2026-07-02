#!/bin/bash
# Run on the HEAD node. Launches `vllm serve` (TP=4, EAGLE3) inside the head
# container in a tmux session so it survives SSH teardown. Logs to ~/m3-serve.log.
set -euo pipefail
NAME=m3-tp4
tmux kill-session -t m3serve 2>/dev/null || true
: > "$HOME/m3-serve.log"
tmux new-session -d -s m3serve "docker exec $NAME vllm serve cyankiwi/MiniMax-M3-AWQ-INT4 \
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
  --speculative-config '{\"method\":\"eagle3\",\"model\":\"Inferact/MiniMax-M3-EAGLE3\",\"num_speculative_tokens\":2,\"attention_backend\":\"TRITON_ATTN\"}' \
  > $HOME/m3-serve.log 2>&1; echo SERVE_EXIT=\$? >> $HOME/m3-serve.log"
sleep 3
echo "serve launched in tmux 'm3serve'; log: ~/m3-serve.log"
tmux ls
