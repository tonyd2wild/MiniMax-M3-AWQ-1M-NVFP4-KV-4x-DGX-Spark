#!/bin/bash
# fix-minimax-m3-nvfp4-kv — nvfp4 (4-bit) KV cache for MiniMax M3 on GB10/sm121.
# Phase 1: main KV of both attention paths (sparse + full) packed nvfp4;
# indexer side cache stays bf16. Scratch-dequant read (correctness-first).
#
# Apply INSIDE the image build for minimax-m3-awq:tp4-sm121-nvfp4kv (never
# against the live m3-tp4 container). Serve with:  --kv-cache-dtype nvfp4
set -euo pipefail

SITE="${SITE_PACKAGES:-/opt/env/lib/python3.12/site-packages}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

if [ ! -d "$SITE/vllm" ]; then
  echo "[m3-nvfp4-kv] ERROR: vllm not found at $SITE (set SITE_PACKAGES)" >&2
  exit 1
fi

echo "[m3-nvfp4-kv] installing nvfp4 KV mod into $SITE"

install() {  # install <mod-file> <site-relative-target>
  local src="$HERE/$1" dst="$SITE/$2"
  if [ -f "$dst" ] && [ ! -f "$dst.orig-nvfp4kv" ]; then
    cp "$dst" "$dst.orig-nvfp4kv"
  fi
  cp "$src" "$dst"
  "$PY" - "$dst" <<'PYEOF'
import ast, sys
ast.parse(open(sys.argv[1]).read())
print(f"[m3-nvfp4-kv] syntax OK: {sys.argv[1]}")
PYEOF
}

install nvfp4_kv.py         vllm/models/minimax_m3/common/ops/nvfp4_kv.py
install sparse_attention.py vllm/models/minimax_m3/common/sparse_attention.py
install indexer.py          vllm/models/minimax_m3/common/indexer.py
install model.py            vllm/models/minimax_m3/nvidia/model.py
install triton_attn.py      vllm/v1/attention/backends/triton_attn.py

# Sanity: the flashinfer in this image must ship the nvfp4 KV quant helpers
# (the same ones the MiMo nvfp4-kv recipe used, sm121-proven).
"$PY" - <<'PYEOF'
import flashinfer
missing = [n for n in ("nvfp4_kv_quantize", "nvfp4_kv_dequantize")
           if not hasattr(flashinfer, n)]
if missing:
    raise SystemExit(
        f"[m3-nvfp4-kv] ERROR: flashinfer lacks {missing} — this mod needs "
        "the flashinfer build used by the MiMo nvfp4-kv recipe.")
print("[m3-nvfp4-kv] flashinfer nvfp4_kv_quantize/dequantize present")
PYEOF

# NOTE: no attention.py monkeypatch needed (unlike the MiMo 0.21 mod): this
# tree's generic Attention layer already resolves nvfp4 -> uint8 storage +
# KVQuantMode.NVFP4 via kv_cache_dtype_str_to_dtype/get_kv_quant_mode.
echo "[m3-nvfp4-kv] done. Serve with --kv-cache-dtype nvfp4"
echo "[m3-nvfp4-kv] (optional) VLLM_M3_NVFP4_ALLOW_CG=1 only under breakable cudagraphs"
