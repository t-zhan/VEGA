#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AUTOVLA_ROOT="${REPO_ROOT}/third_party/AutoVLA"
CONFIG="${AUTOVLA_ROOT}/config/eval/qwen2.5-vl-3B-nusc-eval-local.yaml"
CHECKPOINT="${AUTOVLA_ROOT}/data/autovla_base.ckpt"
SEG_DIR="${AUTOVLA_ROOT}/data/nusc_eval_seg"
NUM_SAMPLES="${1:-2}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate autovla

mkdir -p "${SEG_DIR}"

cd "${AUTOVLA_ROOT}"
# nuplan-devkit 在模块级 import pytest，确保已安装
python -c "import pytest" 2>/dev/null || pip install pytest -q
# 禁止 transformers 加载 TF 后端（protobuf 4.x 与 tensorflow 2.20 不兼容，NuScenes 推理不需要 TF）
export USE_TF=0
echo "Running nusc_eval.py (${NUM_SAMPLES} samples)..."
python tools/eval/nusc_eval.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --seg_data_path "${SEG_DIR}" \
    --num_samples "${NUM_SAMPLES}"
