#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,2,3

CONFIG=training/qwen2.5-vl-3B-nusc-sft-cot-local
CKPT=runs/sft/2026-05-24_19-17-19/epoch=4-loss=0.9184.ckpt
LOG=runs/sft/$(date +%Y-%m-%d_%H-%M-%S)_sft_cot_resume.log

cd "${AUTOVLA_DIR}"
mkdir -p runs/sft
nohup python tools/run_sft.py \
    --config "$CONFIG" \
    --resume "$CKPT" \
    > "$LOG" 2>&1 &
echo "PID: $!"
