#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="autovla_nusc_preprocess"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTOVLA_ROOT="$(cd "${SCRIPT_DIR}/../third_party/AutoVLA" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[1/2] Creating conda environment '${ENV_NAME}' (Python 3.9)..."
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    conda create -n "${ENV_NAME}" python=3.9 -y
fi
conda activate "${ENV_NAME}"
pip install --upgrade pip

echo "[2/2] Installing nuscenes-devkit and dependencies..."
pip install \
    "numpy<2" \
    tqdm \
    pyquaternion \
    torch \
    nuscenes-devkit==1.1.11

echo ""
echo "Done. Activate: conda activate ${ENV_NAME}"
echo "Preprocessing: cd ${AUTOVLA_ROOT} && bash scripts/run_nuscenes_preprocessing.sh --help"
