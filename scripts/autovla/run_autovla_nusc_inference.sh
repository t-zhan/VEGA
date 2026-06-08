#!/usr/bin/env bash

CKPT_DIR="runs/sft/2026-05-24_19-17-19"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p third_party/AutoVLA/runs/logs

_run() {
  local CKPT="$1" DEVICE="$2"
  local EPOCH
  EPOCH=$(basename "${CKPT}" | grep -oP 'epoch=\K[0-9]+')
  local LOG="third_party/AutoVLA/runs/logs/${TIMESTAMP}_eval_epoch${EPOCH}.log"
  nohup bash -c "
    cd third_party/AutoVLA
    python -u tools/eval/nusc_eval.py \
      --config 'config/eval/qwen2.5-vl-3B-nusc-eval-local.yaml' \
      --checkpoint '${CKPT}' \
      --seg_data_path 'data/nusc_eval_seg' \
      --output 'runs/${TIMESTAMP}_eval_epoch${EPOCH}.txt' \
      --device '${DEVICE}'
  " > "${LOG}" 2>&1 &
  echo "epoch${EPOCH} PID=$! DEVICE=${DEVICE} LOG=${LOG}"
}

_run "${CKPT_DIR}/epoch=2-loss=0.9322.ckpt" "cuda:1"
# _run "${CKPT_DIR}/epoch=3-loss=0.9186.ckpt" "cuda:2"
# _run "${CKPT_DIR}/epoch=4-loss=0.9184.ckpt" "cuda:3"

