#!/usr/bin/env bash
set -euo pipefail

# Installs ml-explore/mlx-lm PR #2086 (Inkling 2 Mini) + overlays our M1/M2 Metal kernels & DSpark MTP.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLX_LM_DIR="${1:-$SCRIPT_DIR/mlx-lm}"

if [ ! -d "$MLX_LM_DIR" ]; then
  echo "[1/4] Cloning jundot/mlx-lm (inkling2-mini branch, PR #2086)..."
  git clone --depth=20 --branch inkling2-mini https://github.com/jundot/mlx-lm.git "$MLX_LM_DIR"
else
  echo "[1/4] Using existing $MLX_LM_DIR"
fi

echo "[2/4] Overlaying custom M1/M2 Inkling 2 kernels, fusion, FP16, and DSpark MTP..."
cp "$SCRIPT_DIR/mlx_lm_patch/mlx_lm/models/inkling2/"*.py "$MLX_LM_DIR/mlx_lm/models/inkling2/"

echo "[3/4] Installing editable mlx-lm..."
if command -v uv >/dev/null 2>&1; then
  uv pip install -e "$MLX_LM_DIR"
else
  python3 -m pip install -e "$MLX_LM_DIR"
fi

echo "[4/4] Verifying upstream Inkling 2 unit tests..."
(cd "$MLX_LM_DIR" && python3 -m unittest tests/models/test_inkling2.py -v)
echo "Done! Try: python3 inkling_mlx.py --model ofnordo/Inkling-2-Mini-MLX-W4W8 -p 'Write a Python LRU cache.' -e minimal"
