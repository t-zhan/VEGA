#!/bin/bash
# Batch dim-reduction viz over all parameter combinations.
# Usage:  bash scripts/autovla/batch_embed_viz.sh
set -euo pipefail

export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

embed_viz_jobs=1

h5="${EMBED_OUTPUT_DIR}/train_action_embeddings.h5"


cmds=$(mktemp)
trap "rm -f $cmds" EXIT

for speed in low mid high; do
for lateral in all right straight left; do
for n in 100 1000 10000; do
for ndim in 2 3; do
for method in umap pca tsne; do
    dir="${VIZ_OUTPUT_DIR}/$method"
    out="$dir/${method}_${speed}_${lateral}_${n}_${ndim}d.png"
    [ -f "$out" ] && { echo "Skipping existing: $out"; continue; }
    echo "mkdir -p $dir && python src/visualize/autovla/embed_viz.py --h5 $h5 --speed $speed --lateral $lateral --n $n --method $method --ndim $ndim --output $out" >> "$cmds"
done
done
done
done
done

echo "Running $(wc -l < "$cmds") jobs with ${embed_viz_jobs} workers..."
xargs -0 -a <(tr '\n' '\0' < "$cmds") -P "${embed_viz_jobs}" -I {} bash -c '{}'
echo "Done."