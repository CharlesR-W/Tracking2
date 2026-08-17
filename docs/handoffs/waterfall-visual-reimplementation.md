# Waterfall visual reimplementation

Status: Active

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

## Current work

The archived builders and all three talk GIFs have been inspected. Input schema
and availability are being pinned before implementing the new builder.

## Validation pending

- Unit tests for extraction, hashes, fixed axes, output timing, and deterministic
  regeneration.
- Headless desktop/mobile viewer render with JavaScript-error capture.
- Visual inspection of first, middle, and final frames for every GIF.
- Full publication verifier and layout tests after integration.

## Next action

Add the pinned input manifest and shared artifact-to-visual payload builder.
