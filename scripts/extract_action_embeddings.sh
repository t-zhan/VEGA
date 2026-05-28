#!/usr/bin/env bash
# Extract AutoVLA action token embeddings (first layer + last layer).
# Edit the variables below to configure, then run:
#   bash scripts/extract_action_embeddings.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AUTOVLA_ROOT="${REPO_ROOT}/third_party/AutoVLA"

CONFIG="${AUTOVLA_ROOT}/config/training/qwen2.5-vl-3B-nusc-sft-cot-local.yaml"
CKPT="${AUTOVLA_ROOT}/runs/sft/2026-05-24_19-17-19/epoch=4-loss=0.9184.ckpt"

cd "${AUTOVLA_ROOT}"

# train split on GPU 1
USE_TF=0 conda run -n autovla --no-capture-output python "${REPO_ROOT}/src/extraction/autovla/extract.py" \
    --config "${CONFIG}" --ckpt "${CKPT}" --split train \
    --output "${REPO_ROOT}/outputs/autovla/embedding/train_action_embeddings.npz" \
    --device cuda:1 \
    > "${AUTOVLA_ROOT}/runs/extract_train.log" 2>&1 &

# val split on GPU 2
USE_TF=0 conda run -n autovla --no-capture-output python "${REPO_ROOT}/src/extraction/autovla/extract.py" \
    --config "${CONFIG}" --ckpt "${CKPT}" --split val \
    --output "${REPO_ROOT}/outputs/autovla/embedding/val_action_embeddings.npz" \
    --device cuda:2 \
    > "${AUTOVLA_ROOT}/runs/extract_val.log" 2>&1 &

wait
echo "Both extraction jobs complete."
