#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AUTOVLA_ROOT="${REPO_ROOT}/third_party/AutoVLA"
NUSCENES_SRC="/home/autodrive/Projects/PublicDatasets/NuScenes"
DATA_DIR="${AUTOVLA_ROOT}/data"
OUTPUT_VAL="${DATA_DIR}/nuscenes_val"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[1/3] Setting up data directory and NuScenes symlink..."
mkdir -p "${DATA_DIR}"
if [[ ! -e "${DATA_DIR}/nuscenes" ]]; then
    ln -s "${NUSCENES_SRC}" "${DATA_DIR}/nuscenes"
    echo "  Symlink created: ${DATA_DIR}/nuscenes -> ${NUSCENES_SRC}"
else
    echo "  Symlink already exists: ${DATA_DIR}/nuscenes"
fi
mkdir -p "${DATA_DIR}/nusc_eval_seg"

echo ""
echo "[2/3] Preprocessing NuScenes val split (autovla_nusc_preprocess env)..."
conda activate autovla_nusc_preprocess
cd "${AUTOVLA_ROOT}"
python tools/preprocessing/nusc_sample_generation.py \
    --nuscenes_path "${DATA_DIR}/nuscenes" \
    --output_dir "${OUTPUT_VAL}" \
    --split val \
    --version v1.0-trainval

echo ""
echo "[3/3] Done."
echo "  JSON 数据: ${OUTPUT_VAL}/"
echo "  样本数量: $(find "${OUTPUT_VAL}" -name '*.json' | wc -l)"
