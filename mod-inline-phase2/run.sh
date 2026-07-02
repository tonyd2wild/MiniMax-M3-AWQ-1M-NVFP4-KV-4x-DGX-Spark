#!/bin/bash
# fix-minimax-m3-nvfp4-kv — Phase 2: INLINE-dequant read path (+ optional WMMA
# tensor-core flash-decode) for the nvfp4 (4-bit) KV cache of MiniMax M3 on
# GB10/sm121.  Supersedes the Phase-1 scratch-dequant mod (same store path &
# packed layout; only the READ side changed — the Phase-1 scratch read remains
# available via VLLM_M3_NVFP4_INLINE=0).
#
# Apply INSIDE the image build for minimax-m3-awq:tp4-sm121-nvfp4kv (never
# against the live m3-tp4 container). Serve with:  --kv-cache-dtype nvfp4
set -euo pipefail

SITE="${SITE_PACKAGES:-/opt/env/lib/python3.12/site-packages}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

if [ ! -d "$SITE/vllm" ]; then
  echo "[m3-nvfp4-inline] ERROR: vllm not found at $SITE (set SITE_PACKAGES)" >&2
  exit 1
fi

echo "[m3-nvfp4-inline] installing nvfp4 inline-dequant KV mod into $SITE"

install() {  # install <mod-file> <site-relative-target>
  local src="$HERE/$1" dst="$SITE/$2"
  if [ -f "$dst" ] && [ ! -f "$dst.orig-nvfp4kv" ]; then
    cp "$dst" "$dst.orig-nvfp4kv"
  fi
  cp "$src" "$dst"
  "$PY" - "$dst" <<'PYEOF'
import ast, sys
ast.parse(open(sys.argv[1]).read())
print(f"[m3-nvfp4-inline] syntax OK: {sys.argv[1]}")
PYEOF
}

install nvfp4_kv.py                  vllm/models/minimax_m3/common/ops/nvfp4_kv.py
install sparse_attn.py               vllm/models/minimax_m3/common/ops/sparse_attn.py
install sparse_attention.py          vllm/models/minimax_m3/common/sparse_attention.py
install indexer.py                   vllm/models/minimax_m3/common/indexer.py
install model.py                     vllm/models/minimax_m3/nvidia/model.py
install triton_attn.py               vllm/v1/attention/backends/triton_attn.py
install triton_unified_attention.py  vllm/v1/attention/ops/triton_unified_attention.py
install wmma_decode_m3.py            vllm/v1/attention/ops/wmma_decode_m3.py

# Sanity: flashinfer must ship the nvfp4 KV quant helpers (store path uses
# nvfp4_kv_quantize; the Phase-1 scratch fallback also needs dequantize).
"$PY" - <<'PYEOF'
import flashinfer
missing = [n for n in ("nvfp4_kv_quantize", "nvfp4_kv_dequantize")
           if not hasattr(flashinfer, n)]
if missing:
    raise SystemExit(
        f"[m3-nvfp4-inline] ERROR: flashinfer lacks {missing} — this mod needs "
        "the flashinfer build used by the MiMo nvfp4-kv recipe.")
print("[m3-nvfp4-inline] flashinfer nvfp4_kv_quantize/dequantize present")
PYEOF

echo "[m3-nvfp4-inline] done. Serve with --kv-cache-dtype nvfp4"
echo "[m3-nvfp4-inline] knobs: VLLM_M3_NVFP4_INLINE (default 1=inline; 0=Phase-1 scratch)"
echo "[m3-nvfp4-inline]        VLLM_M3_WMMA_DECODE (default 0; 1=experimental WMMA decode,"
echo "[m3-nvfp4-inline]          measured SLOWER than Triton inline on M3 TP=4 shapes)"
echo "[m3-nvfp4-inline]        VLLM_M3_WMMA_NSPLIT / VLLM_M3_WMMA_MAX_BATCH (WMMA tuning)"
