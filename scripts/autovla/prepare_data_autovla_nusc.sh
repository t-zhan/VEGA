#!/usr/bin/env bash
# 从 VEGA 项目根目录执行: bash scripts/autovla/prepare_data_autovla_nusc.sh

AUTOVLA_ROOT="third_party/AutoVLA"
NUSCENES_SRC="/home/autodrive/Projects/PublicDatasets/NuScenes"
DATA_DIR="${AUTOVLA_ROOT}/data"
SPLIT="train"        # train 或 val
NUM_WORKERS=$(nproc) # 1: 单线程原始脚本; >1: 多线程并行脚本

OUTPUT_DIR="${DATA_DIR}/nuscenes_${SPLIT}"

mkdir -p "${DATA_DIR}"
[[ ! -e "${DATA_DIR}/nuscenes" ]] && ln -s "${NUSCENES_SRC}" "${DATA_DIR}/nuscenes"
mkdir -p "${DATA_DIR}/nusc_eval_seg" "${OUTPUT_DIR}"

cd "${AUTOVLA_ROOT}"

if [[ "${NUM_WORKERS}" -gt 1 ]]; then
    SCRIPT="nusc_sample_generation_parallel.py"
    WORKERS_ARG="--num_workers ${NUM_WORKERS}"
else
    SCRIPT="nusc_sample_generation.py"
    WORKERS_ARG=""
fi

python "tools/preprocessing/${SCRIPT}" \
    --nuscenes_path "data/nuscenes" \
    --output_dir "data/nuscenes_${SPLIT}" \
    --split "${SPLIT}" \
    --version v1.0-trainval \
    --drivelm_path "data/v1_1_train_nus.json" \
    ${WORKERS_ARG}

echo "样本数量: $(find "data/nuscenes_${SPLIT}" -name '*.json' | wc -l)"
