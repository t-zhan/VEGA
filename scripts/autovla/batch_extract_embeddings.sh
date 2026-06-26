#!/usr/bin/env bash
# Extract AutoVLA action token embeddings to HDF5 for multiple checkpoints.
set -euo pipefail
# Uses FIFO-based device pool for multi-GPU parallel execution.
# Activate autovla conda env first, then run from VEGA repo root:
#   conda activate autovla
#   bash scripts/autovla/batch_extract_action_embeddings.sh

split=val
background=false
mode=autoregressive # teacher-forcing / autoregressive
gpu_rank_list="1 2 3"

config="${AUTOVLA_DIR}/config/training/qwen2.5-vl-7B-nusc-sft-cot-local.yaml"
ckpt_dir="${AUTOVLA_DIR}/runs/sft/qwen2.5-7B"

mkdir -p "${EMBED_OUTPUT_DIR}" "${PROJECT_RUNS_DIR}"
cd "${AUTOVLA_DIR}"

# ---- device pool ----
pool="/tmp/device_pool_$$"
mkfifo "${pool}"
exec 3<>"${pool}"
rm "${pool}"
for rank in ${gpu_rank_list}; do
    echo "${rank}" >&3
done
# ----------------------

run_extract() {
    local ckpt="$1"
    local device="$2"
    local out_file="$3"
    local log_file="${4:-}"

    python "${PROJECT_DIR}/src/extraction/autovla/extract.py" \
        --mode "${mode}" \
        --config "${config}" --ckpt "${ckpt}" --split "${split}" \
        --output "${out_file}" \
        --device "${device}" \
        ${log_file:+> "${log_file}" 2>&1}
}

extract_one() {
    local ckpt="$1"
    local device="cuda:${2}"
    local ckpt_name
    ckpt_name="$(basename "${ckpt}" .ckpt | sed 's/=/_/g')"
    local out_file="${EMBED_OUTPUT_DIR}/${split}-${ckpt_name}-${mode}-embeddings.h5"

    if [[ "${background}" == "true" ]]; then
        local timestamp
        timestamp="$(date '+%Y%m%d_%H%M%S')"
        local log_file="${PROJECT_RUNS_DIR}/${timestamp}_extract_${ckpt_name}_${split}.log"
        echo "  [${split}] log: ${log_file}"
        run_extract "${ckpt}" "${device}" "${out_file}" "${log_file}"
    else
        run_extract "${ckpt}" "${device}" "${out_file}"
    fi
}

for ckpt in "${ckpt_dir}"/*.ckpt; do
    read -u 3 rank
    (
        echo "=== $(basename "${ckpt}") on cuda:${rank} ==="
        extract_one "${ckpt}" "${rank}" &
        wait
        echo "${rank}" >&3
    ) &
done
wait

exec 3>&-
echo "All done."
