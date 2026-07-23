# Part E local artifacts

## Canonical checkpointed gate

The canonical Part E gate run is `gate_seed0_20260722/`.

- Model: scratch-trained GPT-2 small on deterministic GPT-2-tokenized TinyStories.
- Training: seed 0, 50M loss tokens.
- Full recovery checkpoints: `checkpoints/tokens_000000000/`,
  `tokens_004000000/`, `tokens_016000000/`, and `tokens_050000000/`.
- Every checkpoint contains `model.safetensors` plus `training_state.pt` with the
  optimizer, CPU/CUDA RNG, exact data-stream RNG, step, token count, and gate
  configuration. Training can therefore resume exactly from any milestone.
- Analysis checkpoints: nine
  `analysis/tokens_*/cut_*/suffix_statistics.json` cells, each independently
  reusable by `tracking2.part_e analyze`.
- Scientific outputs: `suffix_statistics.json`,
  `surrogate_diagnostics.parquet`, and `manifest.json`.
- Provenance: immutable token/replay banks, tokenizer files, source snapshot,
  configuration, logs, and SHA-256 manifests are retained.

All four top-level artifact hashes, four model hashes, and four optimizer-state
hashes were verified after downloading from RunPod on 2026-07-22. Do not delete or
overwrite this directory casually: it is the recovery point for revising Part E
without repeating the full 50M-token training trajectory.

## Interpretation status

This is a measured gate, not a confirmatory result. Minimum PCA coverage was
87.36%, below the preregistered 90% threshold, and projected-true replay incurred
up to 0.0655 nats/token excess loss. The sequence-Gaussian proxy also retained a
substantial and inconsistent true-data excess loss. Keep dashboard Part E marked
as a mockup until the PCA/covariance design is revised and a valid run replaces it.

## Revised checkpoint-only diagnostics (2026-07-22)

Two schema-v2 diagnostics reuse the verified 50M checkpoint without modifying the
canonical gate:

- `diagnostic_v2_seed0_20260722/`: gradient-matched learning rates.
- `diagnostic_v2_equal_lr_seed0_20260722/`: paired equal-learning-rate control.

The revised analyzer uses adaptive PCA (up to 512 components, target coverage
95%), projected-true and token-IID Gaussian controls, a one-step flip-flop
matrix-normal covariance fit, fp32 replay parity, and initial-gradient diagnostics.
All three cuts pass the revised mechanical gates: PCA coverage is 95.02--97.52%,
projected-true excess is 0.0077--0.0144 nats/token, and maximum fp32 parity error
is $1.24\times10^{-5}$.

The original large surrogate gap was substantially amplified by gradient scale:
Gaussian initial gradients were 1.19--3.52 times the true-activation gradient.
After gradient matching, update-32 true-data excess is 0.038--0.049 nats/token at
cuts 0/5 and 0.014 at cut 11. Sequence covariance does not beat mean-only at cuts
0 or 5; it helps modestly at cut 11. Thus the implementation is mechanically
repaired, but the broad sequence-Gaussian sufficiency hypothesis remains
unsupported at this checkpoint.
