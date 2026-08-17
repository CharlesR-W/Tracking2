# Waterfall visual reimplementation

Status: Deprecated — completed 2026-08-17; superseded by content commit
`c5d6f0d`, `STATUS.md`, `RESULTS.md`, and `AGENTS.md`.

## Objective

Reimplement the delivered-talk waterfall plots and GIFs from checked-in saved
artifacts. Use ResNet-18 as the canonical explanatory example, keep the CNN as a
secondary comparison, and preserve the distinction between pilot visuals and
the three-seed controlled CNN evidence.

## Scope and completion condition

- One current artifact-reading implementation produces both a self-contained
  interactive HTML viewer and presentation GIF/static derivatives.
- Input paths and SHA-256 hashes are pinned; no training code is imported.
- ResNet-18 leads the viewer and post waterfall section.
- Rebuilt GIF timing, frame count, fixed scales, static fallbacks, and source
  semantics are tested and visually inspected.
- The post, figure manifest, Pages workflow, evidence docs, and publication
  verifier agree on the active outputs.

## Decisions

- The ResNet visualization is the canonical explanatory example, not the
  canonical quantitative evidence. It is one trained model with three surrogate
  redraws per cut.
- The controlled three-seed small-CNN battery remains the quantitative headline.
- The existing ResNet archive contains complete checkpoints at epochs
  0, 1, 5, 20, and 100 across eight cuts.
- The checked-in CNN archive can reproduce the six coarse checkpoints
  0, 1, 5, 10, 20, and 30 across four cuts. Some within-epoch cut artifacts used
  by the old GIF are absent, so the replacement will not pretend those frames
  are reproducible.
- The signed waterfall stacks encode alternative outcomes compactly; they are
  not additive components. The interactive viewer will also expose a
  common-baseline endpoint view for exact comparison.

## Completed implementation

- `src/tracking2/waterfall_visuals.py` is the single artifact-reading
  implementation for the self-contained viewer, three GIFs, and five static
  exports.
- `artifacts/waterfall_visuals/manifest.json` pins 29 measured JSON inputs by
  path and SHA-256. No training code or checkpoint rerun is involved.
- ResNet-18 leads the post and viewer. The CNN depth and relaxation views remain
  secondary comparisons, and their pilot evidence boundary is explicit.
- `LW post/figure_manifest.json`, the Pages workflow, publication verifier, and
  project documentation all identify the same 14 active assets.

## Completed validation

- The full suite passed with 109 tests; the final focused waterfall/layout run
  passed 8 tests.
- Fixed axes, source semantics, frame counts, timing, input hashes, and
  byte-identical regeneration are tested.
- Desktop and mobile headless renders reported three plots and no JavaScript
  errors. The first, middle, and final frames of every GIF were visually
  inspected.
- The committed publication verifier passed the 29 pilot inputs, 8 waterfall
  exports, 14 active figures, byte-identical viewer, and byte-identical
  appendix.
- GitHub Pages run `32051457837` succeeded; live viewer, GIF, and appendix bytes
  matched the tracked assets.

## Next action

Perform the human LessWrong platform preview described in `STATUS.md`; no
waterfall implementation work remains.
