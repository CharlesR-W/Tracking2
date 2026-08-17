# Tracking2 status

## Current goal

The corrected LessWrong publication package is ready for the author to paste
into LessWrong. The post itself has not been published to LessWrong.

## What works

- The measured July 26 CNN control battery, provenance correction sidecar, and
  deterministic HTML appendix pass the committed publication verifier.
- The original post was restored exactly at commit `9218f5d` from `4371d3c`.
- The delivered talk and its GIFs, waterfall plots, and schematics remain
  recoverable in Git and in `LW post/deprecated/`.
- The corrected package was committed as `6079743`, pushed to `main`, and
  deployed successfully by GitHub Pages on 2026-08-17.
- The active post uses nine hash-locked talk-family assets, including all three
  talk GIFs. Rejected redesign assets are isolated under
  `LW post/deprecated/rejected-2026-07-redesign/`.
- Independent reader, evidence, and final verification reviews passed. The
  full suite reports 106 passing tests, and live Pages bytes match the locked
  appendix and figure hashes.

## Publication state

The Markdown source is `LW post/free-body-diagrams-for-neural-networks.md`. The
supporting appendix and assets are live at
<https://charlesr-w.github.io/Tracking2/>. No LessWrong draft or post was created
by this cleanup.

## Next action

Paste the prepared Markdown into LessWrong and perform the platform preview.

## Active handoffs

None. The cleanup handoff is retained as a completed historical record.
