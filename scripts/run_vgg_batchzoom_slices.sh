#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/Tracking2}"
CONDITION="${CONDITION:-native}"
CUTS="${CUTS:-7 8 9 13}"
THREADS="${THREADS:-4}"

cd "$ROOT"
for cut in $CUTS; do
  output="artifacts/vgg_batchzoom/cut${cut}-${CONDITION}"
  if [[ -s "${output}/vgg_suffix_statistics.json" ]]; then
    echo "[skip] ${output}"
    continue
  fi
  OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    python -m tracking2.vgg_suffix_statistics \
      --output "$output" \
      --checkpoint artifacts/criticality/seed3/checkpoint_epoch100.pt \
      --initialization-checkpoint artifacts/criticality/seed3/checkpoint_epoch0.pt \
      --data-root "$ROOT/data" --batch-norm --cuts "$cut" --conditions "$CONDITION" \
      --train-size 10000 --test-size 2000 --batch-size 128 \
      --pca-fit-size 5000 --pca-components 1400 --surrogate-draws 3 \
      --relax-epochs 1 --relax-batch-zoom --seed 3 --device cuda
done
