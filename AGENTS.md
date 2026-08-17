# Tracking2 agent guide

## Start here

Read `STATUS.md` for current work, then use this map:

- `README.md` — project purpose, supported workflows, and documentation map.
- `SPEC.md` — research design.
- `ARCHITECTURE.md` — subsystem/data-flow map and where-to-change guidance.
- `RESULTS.md` — trusted results, evidence boundaries, and artifact pointers.
- `LW post/README.md` — LessWrong publication-package layout.
- `LW post/lw_post_reproduction.md` — bounded post experiment protocol.
- `src/tracking2/` — experiments and report builders.
- `scripts/` — verification and run entry points.
- `tests/` — behavior, evidence-contract, and publication-layout tests.

Read a handoff only when `STATUS.md` lists it as active. Files under
`docs/handoffs/deprecated/` are historical audit records, not current instructions.

## Standard commands

Install and test:

```bash
uv sync --extra dev
.venv/bin/python -m pytest -q
```

Verify the checked-in LessWrong package:

```bash
PYTHONPATH=src .venv/bin/python scripts/verify_lw_post_publication.py \
  --profile committed
```

Rebuild its deterministic evidence appendix:

```bash
PYTHONPATH=src .venv/bin/python -m tracking2.post_report \
  artifacts/lw_post/dashboard_manifest.json \
  --output "LW post/free-body-diagrams-for-neural-networks.html"
```

Use the more specific run commands in `README.md` and the reproduction notes
only when the corresponding experiment is in scope.

## Publication invariants

- The current post is `LW post/free-body-diagrams-for-neural-networks.md`.
- `LW post/figure_manifest.json` is the exact active-figure allowlist. Active
  method illustrations are byte-identical delivered-talk copies. Active
  waterfall GIFs/static fallbacks are deterministic outputs of
  `scripts/build_waterfall_visuals.py` from the hash-pinned
  `artifacts/waterfall_visuals/manifest.json` inputs.
- The July 2026 redesign is rejected. Nothing under
  `LW post/deprecated/rejected-2026-07-redesign/` may be promoted into the post,
  Pages workflow, or current documentation without an explicit new user decision.
- Keep the two evidence layers distinct: the GIF/waterfall views are historical
  pilots; the July 26 three-seed CNN battery is the controlled quantitative
  evidence. `RESULTS.md` records the permitted claims and non-claims.
- ResNet-18 is the canonical explanatory waterfall example, not independent-seed
  or canonical quantitative evidence. Do not relabel its three redraws as model
  replications.
- `artifacts/lw_post/dashboard_manifest.json` and
  `artifacts/lw_post/provenance_corrections.json` pin the controlled evidence.
  Do not edit measured JSON to repair provenance; use the audited correction
  mechanism and rerun the verifier.
- The tracked HTML appendix is generated from the manifest and must remain
  byte-identical to a fresh build. The waterfall viewer and generated exports
  have the same byte-identity requirement. GitHub Pages publishes those two
  HTML surfaces and the figure-manifest allowlist.

## Repository boundaries

- Treat `deprecated/` and every nested `deprecated/` directory as read-only
  history unless a task explicitly concerns archival repair. Compatibility
  shims may import archived code, but archived runners/verifiers are never the
  authority for a current run or publication decision.
- Do not mistake `MOCKUP/`, smoke outputs, planned panels, or historical pilots
  for measured confirmatory evidence. Mockups must remain visibly marked.
- Large checkpoints and local experiment outputs may be intentionally untracked.
  Never stage broad artifact trees or checkpoint files without first inspecting
  the exact paths and repository status.
- Do not import implementations or design choices from sibling Tracking projects
  unless the user explicitly requests a comparison.
- Paid RunPod work follows the global hard-cost rule and the bounded runner specs;
  terminate resources promptly.

## Validation expectations

- Code changes: run focused tests plus the full suite when behavior crosses
  subsystem boundaries.
- Post, figure, manifest, appendix, or Pages changes: run the committed-profile
  publication verifier, `tests/test_publication_layout.py`, and the relevant
  visual builder tests; visually inspect first/middle/final GIF frames and the
  desktop/mobile viewer.
- Evidence changes: preserve source/config identities, artifact hashes, seed as
  the independent unit, and explicit mock/pilot/measured labels. Update
  `RESULTS.md` when trusted claims or their boundaries change.
- Before handoff: run `git diff --check`, update `STATUS.md`, and keep mutable
  progress out of this file.
