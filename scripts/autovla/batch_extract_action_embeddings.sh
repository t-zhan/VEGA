#!/usr/bin/env bash
# Extract AutoVLA action token embeddings to HDF5 for multiple checkpoints.
# Uses FIFO-based device pool for multi-GPU parallel execution.
# Activate autovla conda env first, then run from VEGA repo root:
#   conda activate autovla
#   bash scripts/autovla/batch_extract_action_embeddings.sh
#
# Set BACKGROUND=false to log to stdout instead of files.

REPO_ROOT="$(pwd)"
AUTOVLA_ROOT="${REPO_ROOT}/third_party/AutoVLA"
CONFIG="${AUTOVLA_ROOT}/config/training/qwen2.5-vl-3B-nusc-sft-cot-local.yaml"
CKPT_DIR="${AUTOVLA_ROOT}/runs/sft/2026-05-24_19-17-19"
OUT_DIR="${REPO_ROOT}/outputs/autovla/embeddings"
LOG_DIR="${REPO_ROOT}/runs"
DEVICES=("cuda:1" "cuda:1" "cuda:2")
BACKGROUND=true

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${AUTOVLA_ROOT}"

# ---- device pool ----
POOL="/tmp/device_pool_$$"
mkfifo "${POOL}"
exec 3<>"${POOL}"
rm "${POOL}"
for dev in "${DEVICES[@]}"; do
    echo "${dev}" >&3
done
# ----------------------

extract_one() {
    local ckpt="$1"
    local split="$2"
    local device="$3"
    local ckpt_name
    ckpt_name="$(basename "${ckpt}" .ckpt | sed 's/=/_/g')"
    local out_file="${OUT_DIR}/${split}-${ckpt_name}-action_text_embeddings.h5"

    if [[ "${BACKGROUND}" == "true" ]]; then
        local timestamp
        timestamp="$(date '+%Y%m%d_%H%M%S')"
        local log_file="${LOG_DIR}/${timestamp}_extract_${ckpt_name}_${split}.log"
        echo "  [${split}] log: ${log_file}"
        python "${REPO_ROOT}/src/extraction/autovla/extract.py" \
            --config "${CONFIG}" --ckpt "${ckpt}" --split "${split}" \
            --output "${out_file}" \
            --device "${device}" \
            > "${log_file}" 2>&1
    else
        python "${REPO_ROOT}/src/extraction/autovla/extract.py" \
            --config "${CONFIG}" --ckpt "${ckpt}" --split "${split}" \
            --output "${out_file}" \
            --device "${device}"
    fi
}

for ckpt in "${CKPT_DIR}"/*.ckpt; do
    read -u 3 dev
    (
        echo "=== $(basename "${ckpt}") on ${dev} ==="
        extract_one "${ckpt}" train "${dev}" &
        # extract_one "${ckpt}" val   "${dev}" &
        wait
        echo "${dev}" >&3
    ) &
done
wait

exec 3>&-
echo "All done."
