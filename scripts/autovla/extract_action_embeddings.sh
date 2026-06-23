#!/usr/bin/env bash
# Extract AutoVLA action token embeddings to HDF5.
# Activate autovla conda env first, then run from VEGA repo root:
#   conda activate autovla
#   bash scripts/autovla/extract_action_embeddings.sh

CONFIG="${AUTOVLA_DIR}/config/training/qwen2.5-vl-3B-nusc-sft-cot-local.yaml"
CKPT="${AUTOVLA_DIR}/runs/sft/2026-05-24_19-17-19/epoch=2-loss=0.9322.ckpt"
mkdir -p "${EMBED_OUTPUT_DIR}" "${PROJECT_RUNS_DIR}"
cd "${AUTOVLA_DIR}"

extract() {
    local split="$1"
    local device="$2"
    local timestamp
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    local out_name="${split}-${MODE}-$(basename "${CKPT}" .ckpt | sed 's/=/_/g')-action_text_embeddings.h5"
    nohup python "${PROJECT_DIR}/src/extraction/autovla/extract.py" \
        --mode "${MODE}" \
        --config "${CONFIG}" --ckpt "${CKPT}" --split "${split}" \
        --output "${EMBED_OUTPUT_DIR}/${out_name}" \
        --device "${device}" \
        --end 100 \
        > "${PROJECT_RUNS_DIR}/${timestamp}_extract_${MODE}_${split}.log" 2>&1 &
}

extract train cuda:2
extract val   cuda:3