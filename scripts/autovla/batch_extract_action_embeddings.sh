#!/usr/bin/env bash
# Extract AutoVLA action token embeddings to HDF5.
# Activate autovla conda env first, then run from VEGA repo root:
#   conda activate autovla
#   bash scripts/autovla/extract_action_embeddings.sh

REPO_ROOT="$(pwd)"
AUTOVLA_ROOT="${REPO_ROOT}/third_party/AutoVLA"
CONFIG="${AUTOVLA_ROOT}/config/training/qwen2.5-vl-3B-nusc-sft-cot-local.yaml"
CKPT_DIR="${AUTOVLA_ROOT}/runs/sft/2026-05-31_07-09-24"
OUT_DIR="${REPO_ROOT}/outputs/autovla/embeddings"
mkdir -p "${OUT_DIR}"
cd "${AUTOVLA_ROOT}"

for ckpt in "${CKPT_DIR}"/*.ckpt; do
    echo "=== Processing: $(basename "${ckpt}") ==="
    for split in train val; do
        python "${REPO_ROOT}/src/extraction/autovla/extract.py" \
            --config "${CONFIG}" --ckpt "${ckpt}" --split "${split}" \
            --output "${OUT_DIR}/${split}_$(basename "${ckpt}" .ckpt | sed 's/=/_/g')-action_embeddings.h5" \
            --device cuda:0
    done
    echo
done
