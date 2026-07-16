#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Tracking2

SEED="${SEED:-6}"
DRAW_COUNT="${DRAW_COUNT:-3}"
PCA_COMPONENTS="${PCA_COMPONENTS:-512}"
ROOT="artifacts/vgg_checkpoints/seed${SEED}"

if [[ ! -f "$ROOT/checkpoint_epoch100.pt" ]]; then
  python -m tracking2.criticality \
    --output "$ROOT" --training-only \
    --train-size 50000 --test-size 10000 --epochs 100 --batch-size 128 \
    --batch-norm --checkpoints 0 1 5 20 100 --lr-milestones 30 60 90 \
    --seed "$SEED" --device cuda
fi

for checkpoint_epoch in 0 1 5 20 100; do
  output="artifacts/vgg_suffix_statistics/seed${SEED}-epoch${checkpoint_epoch}"
  if [[ -f "$output/vgg_suffix_statistics.json" ]]; then
    echo "[skip] $output already complete"
    continue
  fi
  python -m tracking2.vgg_suffix_statistics \
    --output "$output" \
    --checkpoint "$ROOT/checkpoint_epoch${checkpoint_epoch}.pt" \
    --checkpoint-epoch "$checkpoint_epoch" \
    --batch-norm --cuts 7 8 9 10 --conditions native \
    --train-size 10000 --test-size 2000 --batch-size 128 \
    --pca-fit-size 5000 --pca-components "$PCA_COMPONENTS" \
    --surrogate-draws "$DRAW_COUNT" --relax-epochs 5 --relax-batch-zoom \
    --seed "$SEED" --device cuda
done
