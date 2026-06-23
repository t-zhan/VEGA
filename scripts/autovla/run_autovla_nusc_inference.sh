#!/usr/bin/env bash

set -euo pipefail
ckpt_dir="runs/sft/2026-05-24_19-17-19"
timestamp=$(date +%Y%m%d_%H%M%S)

mkdir -p "${AUTOVLA_DIR}/runs/logs"

_run() {
  local ckpt="$1" device="$2"
  local epoch
  epoch=$(basename "${ckpt}" | grep -oP 'epoch=\K[0-9]+')
  local log="${AUTOVLA_DIR}/runs/logs/${timestamp}_eval_epoch${epoch}.log"
  nohup bash -c "
    cd ${AUTOVLA_DIR}
    python -u tools/eval/nusc_eval.py \
      --config 'config/eval/qwen2.5-vl-3B-nusc-eval-local.yaml' \
      --checkpoint '${ckpt}' \
      --seg_data_path 'data/nusc_eval_seg' \
      --output 'runs/${timestamp}_eval_epoch${epoch}.txt' \
      --device '${device}'
  " > "${log}" 2>&1 &
  echo "epoch${epoch} PID=$! device=${device} log=${log}"
}

_run "${ckpt_dir}/epoch=2-loss=0.9322.ckpt" "cuda:1"
# _run "${ckpt_dir}/epoch=3-loss=0.9186.ckpt" "cuda:2"
# _run "${ckpt_dir}/epoch=4-loss=0.9184.ckpt" "cuda:3"

