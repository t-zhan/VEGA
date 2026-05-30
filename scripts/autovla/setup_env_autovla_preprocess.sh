#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="autovla_nusc_preprocess"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[1/2] Creating conda environment '${ENV_NAME}' (Python 3.9)..."
if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    conda create -n "${ENV_NAME}" python=3.9 -y
fi
conda activate "${ENV_NAME}"

echo "[2/3] Downgrading pip to a Python 3.9-compatible version..."
conda install -n "${ENV_NAME}" pip=24.0 -y
conda activate "${ENV_NAME}"

echo "[3/3] Installing nuscenes-devkit and dependencies..."
pip install \
    "numpy<2" \
    tqdm \
    pyquaternion \
    torch \
    nuscenes-devkit==1.1.11

echo ""
echo "Done. Activate: conda activate ${ENV_NAME}"
echo "Preprocessing: cd third_party/AutoVLA && bash scripts/run_nuscenes_preprocessing.sh --help"
