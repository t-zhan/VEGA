#!/usr/bin/env bash
set -euo pipefail

cd /home/tongzhan/Projects/VEGA.worktrees/ARIS/third_party/AutoVLA

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u tools/eval/nusc_eval.py \
  --config config/eval/qwen2.5-vl-3B-nusc-eval-local.yaml \
  --checkpoint pretrained/AutoVLA_PDMS_89.ckpt \
  --seg_data_path data/nusc_eval_seg \
  --output runs/autovla_pdms89_nusc_eval.txt \
  --device cuda:1
