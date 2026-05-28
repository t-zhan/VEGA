#!/usr/bin/env bash

CHECKPOINT="runs/sft/2026-05-24_19-17-19/epoch=4-loss=0.9184.ckpt"
EPOCH=$(basename "${CHECKPOINT}" | grep -oP 'epoch=\K[0-9]+')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="third_party/AutoVLA/runs/${TIMESTAMP}_eval_epoch${EPOCH}.log"

mkdir -p third_party/AutoVLA/runs

nohup bash -c "
  cd third_party/AutoVLA
  python -u tools/eval/nusc_eval.py \
    --config 'config/eval/qwen2.5-vl-3B-nusc-eval-local.yaml' \
    --checkpoint '${CHECKPOINT}' \
    --seg_data_path 'data/nusc_eval_seg' \
    --output 'runs/${TIMESTAMP}_eval_epoch${EPOCH}.txt' \
    --device 'cuda:3'
" > "${LOG}" 2>&1 &

echo "PID=$! LOG=${LOG}"
