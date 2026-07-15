#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Tracking2
python -m pip install -e .
mkdir -p artifacts/confirmatory logs

for seed in {0..19}; do
  if [[ -f "artifacts/confirmatory/seed${seed}/results.json" ]]; then
    echo "[skip] seed ${seed} already complete"
    continue
  fi
  python -m tracking2.experiment \
    --output "artifacts/confirmatory/seed${seed}" \
    --train-size 50000 --test-size 10000 \
    --pca-fit-size 20000 --pca-components 128 \
    --epochs 30 --batch-size 256 --widths 32 64 128 128 \
    --architectures residual \
    --train-distributions mean covariance true \
    --checkpoints 0 1 2 5 10 20 30 \
    --refit-steps 100 --refit-batches 12 \
    --fisher-samples 32 --seed "$seed" --device cuda \
    2>&1 | tee "logs/seed${seed}.log"
done

touch artifacts/confirmatory/COMPLETE
