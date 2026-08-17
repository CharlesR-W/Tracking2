# Tracking2 status

## Current state

The corrected LessWrong publication package is ready for the author to paste
into LessWrong. It is committed on `main` and its evidence appendix and assets
are deployed. The post itself has not been published to LessWrong.

## Working and verified

- The post uses the prose lineage restored by `9218f5d` from trusted baseline
  `4371d3c` and nine hash-locked delivered-talk assets, including all three GIFs.
- The rejected July/August redesign is isolated under
  `LW post/deprecated/rejected-2026-07-redesign/` and excluded from the post,
  figure manifest, verifier, and Pages workflow.
- The measured July 26 CNN control battery, audited provenance corrections, and
  deterministic HTML appendix pass the committed publication verifier.
- Independent reader, evidence, and final verification reviews passed. The
  full suite reports 106 passing tests.

## Open work and blockers

Only the LessWrong platform preview and publication remain. There is no code or
evidence blocker. The GIF/waterfall figures are historical pilots rather than
independent-seed evidence; `RESULTS.md` records the controlled claim boundary.

## Next action

Paste the prepared Markdown into LessWrong and perform the platform preview.

## Active handoffs

None. The cleanup handoff is retained as a deprecated historical record.

## Where to look

- Post: `LW post/free-body-diagrams-for-neural-networks.md`
- Evidence and claim boundaries: `RESULTS.md`
- Package and asset map: `LW post/README.md`
- Live appendix and figures: <https://charlesr-w.github.io/Tracking2/>
- Deprecated cleanup record:
  `docs/handoffs/deprecated/lw-post-trusted-lineage-cleanup.md`

## Last verified

2026-08-17 for publication content commit `6079743`: 106 tests and the
committed-profile publication verifier passed; three independent reviews passed.
The documentation-only deployment at `6c0bdd5` (Pages run `32047025413`) also
succeeded, and the live appendix matched the tracked SHA-256
`2d26a06bfeaeba100f245cded71482abd4710466d093c98ab4c3854eaa8dee65`.
