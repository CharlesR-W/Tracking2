# Waterfall visual inputs

`manifest.json` is the complete, hash-pinned input contract for the current
ResNet-first waterfall explorer and its GIF/static exports. All 29 inputs are
saved historical pilot JSON files under `deprecated/2026-07-omnibus/artifacts/`.
The builder reads them without importing training code:

```bash
PYTHONPATH=src .venv/bin/python scripts/build_waterfall_visuals.py
```

The outputs are `LW post/waterfalls.html` and the eight generated assets named
by `LW post/figure_manifest.json`. The strict publication verifier rebuilds all
of them in a temporary directory and requires byte identity.

Evidence boundaries:

- ResNet-18 is one trained model observed at five checkpoints and eight cuts.
  Each cell has three surrogate redraws; those are not independent model seeds.
- The CNN depth frames are separately scheduled models at six coarse
  checkpoints, not one training trajectory.
- The CNN relaxation animation fixes one epoch-1 checkpoint and one cut and
  reveals genuine suffix-relaxation epochs 0 through 10.
- Some within-epoch CNN artifacts used by the delivered-talk GIF are not
  checked in across all four cuts. The reproducible replacement deliberately
  omits those frames.

These pilots explain the experiment. They are not part of the controlled
three-seed CNN evidence manifest under `artifacts/lw_post/`.
