#!/usr/bin/env bash
set -euo pipefail

h5_files=(
  "${EMBED_OUTPUT_DIR}/train-epoch_15-loss_0.9363-teacher-forcing-embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_16-loss_0.9271-teacher-forcing-embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_17-loss_0.9279-teacher-forcing-embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_18-loss_0.9201-teacher-forcing-embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_19-loss_0.9167-teacher-forcing-embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_20-loss_0.9224-teacher-forcing-embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_21-loss_0.9110-teacher-forcing-embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_22-loss_0.9103-teacher-forcing-embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_23-loss_0.9104-teacher-forcing-embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_24-loss_0.9128-teacher-forcing-embeddings.h5"
  "${EMBED_OUTPUT_DIR}/train-epoch_25-loss_0.9021-teacher-forcing-embeddings.h5"
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
