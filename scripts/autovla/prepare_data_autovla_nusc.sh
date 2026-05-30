#!/usr/bin/env bash
# 从 VEGA 项目根目录执行: bash scripts/autovla/prepare_data_autovla_nusc.sh
set -euo pipefail

cd "third_party/AutoVLA"

preprocess_nusc() {
    local SPLIT="${1:-train}"
    local NUM_WORKERS="${2:-$(nproc)}"
    local OUTPUT_DIR="data/nuscenes_${SPLIT}"

    mkdir -p "${OUTPUT_DIR}"

    if [[ "${NUM_WORKERS}" -gt 1 ]]; then
        local SCRIPT="nusc_sample_generation_parallel.py"
        local WORKERS_ARG="--num_workers ${NUM_WORKERS}"
    else
        local SCRIPT="nusc_sample_generation.py"
        local WORKERS_ARG=""
    fi

    # DriveLM annotations only apply to training data
    local DRIVELM_ARG=""
    if [[ "${SPLIT}" == "train" && -f "data/v1_1_train_nus.json" ]]; then
        DRIVELM_ARG="--drivelm_path data/v1_1_train_nus.json"
    fi

    python "tools/preprocessing/${SCRIPT}" \
        --nuscenes_path "data/nuscenes" \
        --output_dir "${OUTPUT_DIR}" \
        --split "${SPLIT}" \
        --version v1.0-trainval \
        ${DRIVELM_ARG} \
        ${WORKERS_ARG}

    echo "样本数量: $(find "${OUTPUT_DIR}" -name '*.json' | wc -l)"
}

# ── 调用 ─────────────────────────────────────────────────────────
preprocess_nusc "train" "$(nproc)"
preprocess_nusc "val" "$(nproc)"
