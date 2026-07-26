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
DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-artifacts/resnet_criticality/seed${SEED}/checkpoint_epoch100.pt}"
TRAINING_MANIFEST="${TRAINING_MANIFEST:-artifacts/resnet_criticality/seed${SEED}/resnet_training.json}"
RESULT_ROOT="artifacts/lw_post/resnet_ablations_seed${SEED}"
OUTPUT="$RESULT_ROOT/nested_ranks_epoch100_cuts4_8"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing ResNet checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if ! python scripts/validate_architecture_checkpoints.py \
  "$TRAINING_MANIFEST" resnet "$SEED"; then
  echo "Invalid ResNet training manifest: $TRAINING_MANIFEST" >&2
  exit 1
fi

if [[ -f "$OUTPUT/resnet_suffix_statistics.json" ]]; then
  echo "[skip] $OUTPUT"
else
  python -m tracking2.resnet_suffix_statistics \
    --output "$OUTPUT" \
    --checkpoint "$CHECKPOINT" --checkpoint-epoch 100 \
    --training-manifest "$TRAINING_MANIFEST" \
    --data-backend torchvision \
    --cuts 3 7 --train-size 10000 --test-size 2000 --batch-size 128 \
    --pca-fit-size 5000 --pca-ranks 128 256 512 \
    --surrogate-draws 3 --mean-noise-radii 1 \
    --include-projected-true --true-eval-only --relax-epochs 5 \
    --seed "$SEED" --device "$DEVICE"
fi

python scripts/verify_lw_post_resnet_ablations.py \
  "$CHECKPOINT" "$TRAINING_MANIFEST" "$OUTPUT" "$SEED"
