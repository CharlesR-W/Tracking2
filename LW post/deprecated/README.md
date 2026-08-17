# Deprecated LessWrong assets

This directory physically separates preliminary and delivered-talk material
from the current LessWrong package. Nothing here supports the post's current
quantitative claims.

- `talk/` preserves the delivered HTML and PDF exports. The exports are frozen;
  `slides.html` continues to resolve its `../figures/...` assets because the
  talk and historical figures remain sibling directories.
- `figures/` contains the preliminary CNN/ResNet plots, talk illustrations, and
  other figures not used by the current post.
- `notebooks/animate_training_dynamics.py` is the historical animation builder.
  It reads only from the repository archive at
  `../../deprecated/2026-07-omnibus/artifacts/` and writes only to this
  deprecated figure directory.

The current post, appendix, measured-only figure generator, and exactly five
publication figures remain one directory above this archive.
