#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Tracking2
python -m pip install -e .
mkdir -p artifacts/criticality logs

# Run one seed first as the phenomenon gate. Increase MAX_SEEDS only after its
# heatmap shows layer heterogeneity. A caller can wrap this script in `timeout`
# to enforce the remaining RunPod wall-clock budget.
MAX_SEEDS="${MAX_SEEDS:-1}"

for seed in 0 1 2 3 4; do
  if [[ "$seed" -ge "$MAX_SEEDS" ]]; then
    break
  fi
  if [[ -f "artifacts/criticality/seed${seed}/criticality.json" ]]; then
    echo "[skip] seed ${seed} already complete"
    continue
  fi
  python -m tracking2.criticality \
    --output "artifacts/criticality/seed${seed}" \
    --train-size 50000 --test-size 10000 \
    --epochs 100 --batch-size 128 \
    --batch-norm \
    --checkpoints 0 1 2 5 10 20 40 100 \
    --lr-milestones 30 60 90 \
    --recovery-modules 0 4 8 12 18 \
    --recovery-sources random 0 \
    --recovery-steps 200 --recovery-eval-steps 0 10 50 200 \
    --seed "$seed" --device cuda \
    2>&1 | tee "logs/criticality-seed${seed}.log"
done

python scripts/aggregate_criticality.py artifacts/criticality/seed*/criticality.json \
  --output artifacts/criticality/results.json
