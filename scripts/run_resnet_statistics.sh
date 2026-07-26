#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${TRACKING2_SOURCE_REVISION:-}" ]] \
  && git rev-parse HEAD >/dev/null 2>&1 \
  && [[ -z "$(git status --porcelain)" ]]; then
  TRACKING2_SOURCE_REVISION="$(git rev-parse HEAD)"
  export TRACKING2_SOURCE_REVISION
fi
if [[ ! "${TRACKING2_SOURCE_REVISION:-}" =~ ^[0-9a-fA-F]{40}$|^[0-9a-fA-F]{64}$ ]] \
  && [[ ! "${TRACKING2_SOURCE_ARCHIVE_SHA256:-}" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "Measured runs require a clean full TRACKING2_SOURCE_REVISION or TRACKING2_SOURCE_ARCHIVE_SHA256." >&2
  exit 1
fi

SEED="${SEED:-0}"
DRAW_COUNT="${DRAW_COUNT:-3}"
PCA_RANKS="${PCA_RANKS:-512}"
RELAX_EPOCHS="${RELAX_EPOCHS:-5}"
RUN_TAG="${RUN_TAG:-c_gate_v3}"
ROOT="artifacts/resnet_criticality/seed${SEED}"
TRAINING_MANIFEST="$ROOT/resnet_training.json"
read -r -a PCA_RANK_ARGS <<< "$PCA_RANKS"

checkpoints_complete=true
for checkpoint_epoch in 0 1 5 20 100; do
  if [[ ! -f "$ROOT/checkpoint_epoch${checkpoint_epoch}.pt" ]]; then
    checkpoints_complete=false
  fi
done
if ! python scripts/validate_architecture_checkpoints.py \
  "$TRAINING_MANIFEST" resnet "$SEED"; then
  checkpoints_complete=false
fi

if [[ "$checkpoints_complete" != true ]]; then
  python -m tracking2.resnet_criticality \
    --output "$ROOT" --train-size 50000 --test-size 10000 \
    --data-backend torchvision \
    --epochs 100 --batch-size 128 --checkpoints 0 1 5 20 100 \
    --lr-milestones 30 60 90 --weight-decay 0 \
    --seed "$SEED" --device cuda --training-only
fi
python scripts/validate_architecture_checkpoints.py \
  "$TRAINING_MANIFEST" resnet "$SEED"

for checkpoint_epoch in 0 1 5 20 100; do
  output="artifacts/c_architecture_statistics/resnet/${RUN_TAG}-seed${SEED}-epoch${checkpoint_epoch}"
  if [[ -f "$output/resnet_suffix_statistics.json" ]]; then
    echo "[skip] $output already complete"
    continue
  fi
  python -m tracking2.resnet_suffix_statistics \
    --output "$output" --checkpoint "$ROOT/checkpoint_epoch${checkpoint_epoch}.pt" \
    --training-manifest "$TRAINING_MANIFEST" --data-backend torchvision \
    --checkpoint-epoch "$checkpoint_epoch" --cuts 0 1 2 3 4 5 6 7 \
    --train-size 10000 --test-size 2000 --batch-size 128 \
    --pca-fit-size 5000 --pca-ranks "${PCA_RANK_ARGS[@]}" \
    --surrogate-draws "$DRAW_COUNT" --include-projected-true \
    --true-eval-only --relax-epochs "$RELAX_EPOCHS" \
    --seed "$SEED" --device cuda
done
