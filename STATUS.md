# Tracking2 status

## Current state

The corrected LessWrong package is ready for platform preview but remains
unpublished on LessWrong. The talk-era waterfall plots and GIFs have been
reimplemented from pinned saved artifacts, with ResNet-18 as the explanatory
lead and CNN views as secondary comparisons.

## Working and verified

- The post uses the prose lineage restored by `9218f5d` from trusted baseline
  `4371d3c`, six exact delivered-talk method assets, and eight deterministic
  waterfall exports generated from 29 hash-pinned JSON inputs.
- ResNet-18 is the canonical explanatory waterfall: one trained model, eight
  native cuts, five training checkpoints, and three surrogate redraws per cell.
  The six-checkpoint CNN depth view and fixed-checkpoint relaxation view remain
  secondary pilot comparisons.
- The rejected July/August redesign is isolated under
  `LW post/deprecated/rejected-2026-07-redesign/` and excluded from the post,
  figure manifest, verifier, and Pages workflow.
- The measured July 26 CNN control battery, audited provenance corrections, and
  deterministic HTML appendix pass the committed publication verifier. The
  same verifier rebuilds the waterfall viewer and all eight exports
  byte-for-byte.
- Independent reader, evidence, and final verification reviews passed. The
  full suite reports 109 passing tests.

## Open work and blockers

There is no code, evidence, asset-lineage, or deployment blocker. Some
within-epoch CNN artifacts used by the old talk GIF were not checked in, so the
replacement honestly uses the complete six-checkpoint coarse sweep. The
waterfall figures remain historical pilots rather than independent-seed
evidence; `RESULTS.md` records the controlled claim boundary.

## Next action

Paste the Markdown into LessWrong, inspect the platform preview at desktop and
mobile widths, and publish only after that human preview passes.

## Active handoffs

None.

## Where to look

- Post: `LW post/free-body-diagrams-for-neural-networks.md`
- Evidence and claim boundaries: `RESULTS.md`
- Package and asset map: `LW post/README.md`
- Live appendix and figures: <https://charlesr-w.github.io/Tracking2/>
- Deprecated cleanup record:
  `docs/handoffs/deprecated/lw-post-trusted-lineage-cleanup.md`
- Completed waterfall reconstruction record:
  `docs/handoffs/deprecated/waterfall-visual-reimplementation.md`

## Last verified

2026-08-17 for waterfall content commit `c5d6f0d` and Pages dependency fix
`688e3bb`: 109 tests passed, followed by 8 focused layout/waterfall tests and the
committed-profile verifier. Pages run `32051457837` succeeded. The live viewer,
three GIFs, and appendix matched their tracked SHA-256 hashes; the viewer hash is
`1d5b1769d9656a59e09753d0fa0e8cd858d0ece089066977685f081a295cad42`.
