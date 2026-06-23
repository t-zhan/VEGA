#!/usr/bin/env bash
# Extract AutoVLA action token embeddings to HDF5 for multiple checkpoints.
# Uses FIFO-based device pool for multi-GPU parallel execution.
# Activate autovla conda env first, then run from VEGA repo root:
#   conda activate autovla
#   bash scripts/autovla/batch_extract_action_embeddings.sh
#
# Set BACKGROUND=true to write log files instead of stdout.

CONFIG="${AUTOVLA_DIR}/config/training/qwen2.5-vl-7B-nusc-sft-cot-local.yaml"
CKPT_DIR="${AUTOVLA_DIR}/runs/sft/qwen2.5-7B"
GPU_RANK_LIST="1 2 3"
BACKGROUND="${BACKGROUND:-false}"

mkdir -p "${EMBED_OUTPUT_DIR}" "${PROJECT_RUNS_DIR}"
cd "${AUTOVLA_DIR}"

# ---- device pool ----
POOL="/tmp/device_pool_$$"
mkfifo "${POOL}"
exec 3<>"${POOL}"
rm "${POOL}"
for rank in ${GPU_RANK_LIST}; do
    echo "${rank}" >&3
done
# ----------------------

extract_one() {
    local ckpt="$1"
    local split="$2"
    local device="cuda:${3}"
    local ckpt_name
    ckpt_name="$(basename "${ckpt}" .ckpt | sed 's/=/_/g')"
    local out_file="${EMBED_OUTPUT_DIR}/${split}-${ckpt_name}-action_text_embeddings.h5"

    if [[ "${BACKGROUND}" == "true" ]]; then
        local timestamp
        timestamp="$(date '+%Y%m%d_%H%M%S')"
        local log_file="${PROJECT_RUNS_DIR}/${timestamp}_extract_${ckpt_name}_${split}.log"
        echo "  [${split}] log: ${log_file}"
        python "${PROJECT_DIR}/src/extraction/autovla/extract.py" \
            --mode "${MODE}" \
            --config "${CONFIG}" --ckpt "${ckpt}" --split "${split}" \
            --output "${out_file}" \
            --device "${device}" \
            > "${log_file}" 2>&1
    else
        python "${PROJECT_DIR}/src/extraction/autovla/extract.py" \
            --mode "${MODE}" \
            --config "${CONFIG}" --ckpt "${ckpt}" --split "${split}" \
            --output "${out_file}" \
            --device "${device}"
    fi
}

for ckpt in "${CKPT_DIR}"/*.ckpt; do
    read -u 3 rank
    (
        echo "=== $(basename "${ckpt}") on cuda:${rank} ==="
        extract_one "${ckpt}" train "${rank}" &
        # extract_one "${ckpt}" val   "${rank}" &
        wait
        echo "${rank}" >&3
    ) &
done
wait

exec 3>&-
echo "All done."
