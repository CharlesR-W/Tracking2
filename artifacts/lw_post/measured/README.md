# Measured LW-post controls

These JSON files are the measured inputs used by
`artifacts/lw_post/dashboard_manifest.json`. They are deliberately small:
checkpoints and cached activation banks are not committed.

The current evidence surface contains:

- three independently trained CNN seeds for the warm, matched-initial-update,
  rank-2048 comparison after blocks 1 and 4;
- same-checkpoint fixed-learning-rate controls for seed 0;
- a covariance-free PCA adequacy sweep to rank 4096;
- seed-0 exact-versus-shrunk covariance and mean-noise-radius controls;
- seed-0 rank-4096, suffix-reinitialization, and 20-epoch horizon
  sensitivities; and
- the three training manifests needed to audit checkpoint, dataset, and source
  lineage.

Important interpretation limits:

- the three-seed rank-2048 shallow comparison is a projected-subspace
  estimand, because its predictive-KL projection gate fails;
- the exploratory seed-0 rank-4096 cell passes the declared projection gate;
- fixed learning rate and matched initial update are different optimizer
  interventions and must not be pooled;
- warm and reinitialized suffixes must not be pooled;
- mean-only conclusions depend on the declared isotropic-noise radius; and
- moving the cut changes the native suffix's capacity and trainability.

Rebuild the manifest with `scripts/make_lw_post_manifest.py`, then rebuild the
self-contained dashboard with `python -m tracking2.post_report`.
