#!/usr/bin/env bash
set -euo pipefail

H5_FILES=(
  "${EMBED_OUTPUT_DIR}/train-epoch_15-loss_0.9363-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_16-loss_0.9271-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_17-loss_0.9279-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_18-loss_0.9201-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_19-loss_0.9167-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_20-loss_0.9224-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_21-loss_0.9110-action_text_embeddings.h5"
)

running=0
for h5 in "${H5_FILES[@]}"; do
  while (( running >= METRICS_MP_JOBS )); do
    wait -n
    running=$((running - 1))
  done

  (
    echo "=== ${h5} ==="
    python src/analysis/autovla/run_all_metrics_mp.py --h5 "${h5}"
  ) &
  running=$((running + 1))
done

wait
