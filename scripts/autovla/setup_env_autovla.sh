#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="autovla"
AUTOVLA_ROOT="$(pwd)/third_party/AutoVLA"

source "$(conda info --base)/etc/profile.d/conda.sh"

# ── Create env ─────────────────────────────────────────────────
conda create -n "$ENV_NAME" python=3.9 'pip<25' -y
conda activate "$ENV_NAME"

# ── All deps (torch, autoawq, ...) via setup.py ────────────────
pip install -e "$AUTOVLA_ROOT" \
    --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    --extra-index-url https://download.pytorch.org/whl/cu121

# ── flash-attn ─────────────────────────────────────────────────
CXX11ABI=$(python -c "import torch; print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')")
FLASH_FNAME="flash_attn-2.7.4.post1+cu12torch2.4cxx11abi${CXX11ABI}-cp39-cp39-linux_x86_64.whl"
curl -fsSL -o "/tmp/${FLASH_FNAME}" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FLASH_FNAME}"
pip install "/tmp/${FLASH_FNAME}"
rm -f "/tmp/${FLASH_FNAME}"

# ── waymo-open-dataset (patched wheel, --no-deps) ──────────────
WMO_DIR=$(mktemp -d)
pip download waymo-open-dataset-tf-2-12-0==1.6.7 --no-deps -d "$WMO_DIR" -q
WMO_SRC=$(ls "$WMO_DIR"/*.whl | head -1)
WMO_DST="${WMO_SRC%.whl}_patched.whl"
PATCH_DIR=$(mktemp -d)
cp "$WMO_SRC" "$WMO_DST"
META_PATH=$(unzip -l "$WMO_DST" | awk '/METADATA$/{print $NF}')
unzip -q "$WMO_DST" "$META_PATH" -d "$PATCH_DIR"
sed -i 's/\(Requires-Dist: [^=]*\)==/\1>=/g' "$PATCH_DIR/$META_PATH"
(cd "$PATCH_DIR" && zip -q "$WMO_DST" "$META_PATH")
pip install "$WMO_DST" --no-deps
rm -rf "$WMO_DIR" "$PATCH_DIR"

echo "Done. Activate: conda activate ${ENV_NAME}"
