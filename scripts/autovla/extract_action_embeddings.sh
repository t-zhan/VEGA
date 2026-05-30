#!/usr/bin/env bash
# Extract AutoVLA action token embeddings to HDF5.
# Activate autovla conda env first, then run from VEGA repo root:
#   conda activate autovla
#   bash scripts/autovla/extract_action_embeddings.sh

REPO_ROOT="$(pwd)"
AUTOVLA_ROOT="${REPO_ROOT}/third_party/AutoVLA"
CONFIG="${AUTOVLA_ROOT}/config/training/qwen2.5-vl-3B-nusc-sft-cot-local.yaml"
CKPT="${AUTOVLA_ROOT}/runs/sft/2026-05-24_19-17-19/epoch=4-loss=0.9184.ckpt"
OUT_DIR="${REPO_ROOT}/outputs/autovla/embeddings"
LOG_DIR="${REPO_ROOT}/runs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${AUTOVLA_ROOT}"

extract() {
    local split="$1"
    local device="$2"
    nohup python "${REPO_ROOT}/src/extraction/autovla/extract.py" \
        --config "${CONFIG}" --ckpt "${CKPT}" --split "${split}" \
        --output "${OUT_DIR}/${split}_action_embeddings.h5" \
        --device "${device}" \
        > "${LOG_DIR}/extract_${split}.log" 2>&1 &
}

extract train cuda:0
extract val   cuda:0
