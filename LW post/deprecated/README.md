# Deprecated LessWrong assets

This directory preserves historical and rejected LessWrong material. Its
contents have different statuses; consult the relevant README before reuse.

- `talk/` preserves the delivered HTML and PDF exports. The exports are frozen;
  `slides.html` continues to resolve its `../figures/...` assets because the
  talk and historical figures remain sibling directories.
- `figures/` contains the frozen delivered-talk assets and additional pilot
  plots. Exact copies of selected talk-family assets are active in
  `../figures/`; this duplicate is intentional so the frozen talk still works.
- `notebooks/animate_training_dynamics.py` is the historical animation builder.
  It reads only from the repository archive at
  `../../deprecated/2026-07-omnibus/artifacts/` and writes only to this
  deprecated figure directory.
- `rejected-2026-07-redesign/` isolates the unapproved prose/figure redesign
  family that was accidentally promoted in August. Nothing in that directory
  may be treated as a current post asset.

The active post, figure manifest, selected talk-family assets, and controlled
interactive appendix remain one directory above this archive.
