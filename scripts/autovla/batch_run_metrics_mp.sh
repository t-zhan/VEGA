#!/usr/bin/env bash
set -euo pipefail

h5_files=(
  "${EMBED_OUTPUT_DIR}/train-epoch_15-loss_0.9363-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_16-loss_0.9271-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_17-loss_0.9279-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_18-loss_0.9201-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_19-loss_0.9167-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_20-loss_0.9224-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_21-loss_0.9110-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_22-loss_0.9103-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_23-loss_0.9104-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_24-loss_0.9128-action_text_embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_25-loss_0.9021-action_text_embeddings.h5"
)

metrics_mp_jobs=6
running=0

for h5 in "${h5_files[@]}"; do
  while (( running >= metrics_mp_jobs )); do
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
