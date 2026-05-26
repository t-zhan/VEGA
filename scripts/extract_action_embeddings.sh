#!/usr/bin/env bash
# Extract AutoVLA action token embeddings (first layer + last layer).
# Edit the variables below to configure, then run:
#   bash scripts/extract_action_embeddings.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTOVLA_ROOT="${REPO_ROOT}/third_party/AutoVLA"

CONFIG="${AUTOVLA_ROOT}/config/training/qwen2.5-vl-3B-nusc-sft-cot-local.yaml"
CKPT="${AUTOVLA_ROOT}/runs/sft/2026-05-24_19-17-19/epoch=1-loss=0.9526.ckpt"
SPLIT="val"
OUTPUT="${REPO_ROOT}/outputs/embedding/action_embeddings.npz"
DEVICE="cuda"

cd "${AUTOVLA_ROOT}"
conda run -n autovla python "${REPO_ROOT}/src/extraction/extract.py" \
    --config "${CONFIG}" --ckpt "${CKPT}" --split "${SPLIT}" \
    --output "${OUTPUT}" --device "${DEVICE}"
