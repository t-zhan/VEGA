#!/usr/bin/env bash
# 预处理 nuScenes 数据集。
# 用法: bash scripts/autovla/prepare_data_autovla_nusc.sh
set -euo pipefail

cd "${AUTOVLA_DIR}"

preprocess_nusc() {
    local split="$1"
    local num_workers="$2"
    local output_dir="data/nuscenes_${split}"

    mkdir -p "${output_dir}"

    if [[ "${num_workers}" -gt 1 ]]; then
        local script="nusc_sample_generation_parallel.py"
        local workers_arg="--num_workers ${num_workers}"
    else
        local script="nusc_sample_generation.py"
        local workers_arg=""
    fi

    # DriveLM annotations only apply to training data
    local drivelm_arg=""
    if [[ "${split}" == "train" && -f "data/v1_1_train_nus.json" ]]; then
        drivelm_arg="--drivelm_path data/v1_1_train_nus.json"
    fi

    python "tools/preprocessing/${script}" \
        --nuscenes_path "data/nuscenes" \
        --output_dir "${output_dir}" \
        --split "${split}" \
        --version v1.0-trainval \
        ${drivelm_arg} \
        ${workers_arg}

    echo "样本数量: $(find "${output_dir}" -name '*.json' | wc -l)"
}

# ── 调用 ─────────────────────────────────────────────────────────
preprocess_nusc train "$(nproc)"
