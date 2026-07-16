#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/Tracking2}"
DEVICE="${DEVICE:-cpu}"

cd "$ROOT"
for batch in 0 5 20 98; do
  python -m tracking2.suffix_statistics \
    --output "artifacts/suffix_statistics/relax_zoom_batch${batch}_cut3_full_seed0" \
    --checkpoint-epoch 0 --checkpoint-batches "$batch" --cut 3 \
    --train-size 50000 --test-size 10000 --batch-size 256 \
    --pca-fit-size 20000 --pca-components 256 \
    --relax-epochs 1 --relax-batch-zoom --seed 0 --device "$DEVICE"
done
