# Tracking2 publication-ready coordinator handoff

- Audit date: 2026-08-16
- Repository: `/home/crw/Programming/Experiments/Tracking2`
- Current branch: `agent/part-e-dashboard-context`
- Publication target: *Free-body diagrams for neural networks*

## Executive decision

Do **not** launch a GPU rerun for the current LessWrong claim. The canonical CNN
evidence is present, its nine manifest inputs validate, the two measured figures
regenerate byte-for-byte, and the interactive appendix rebuilds byte-for-byte.
No RunPod pod is active, and this audit spent $0.

The package is not publication-ready yet. Its blockers are:

1. the appendix presents fixed learning rate as the primary analysis even though
   the post's headline result uses matched first updates;
2. several provenance fields contain invalid 40-character Git hashes;
3. the radius-zero control is described as first-update matched even though its
   learning-rate multiplier hit the lower clip;
4. the named paid runner/verifier encode an obsolete protocol;
5. the `LW post/` move and publication changes are not committed or merged to
   `main`;
6. final release status and LessWrong asset URLs are undecided; and
7. historical pilot material is labelled but not yet physically separated from
   the publication package.

Treat these as code, metadata, layout, and editorial tasks. A new experiment is
not the default remedy.

## Canonical evidence boundary

Keep these as the publication package:

- post: `LW post/free-body-diagrams-for-neural-networks.md`;
- appendix: `LW post/free-body-diagrams-for-neural-networks.html`;
- evidence index: `artifacts/lw_post/dashboard_manifest.json`;
- nine manifest-listed CNN JSON inputs under `artifacts/lw_post/measured/`;
- three CNN training manifests under
  `artifacts/lw_post/measured/training_manifests/`;
- reproduction notes: `LW post/lw_post_reproduction.md`;
- figure generator: `LW post/notebooks/lw_post_figures.py`; and
- exactly these five post-used figures:
  - `class_conditioned_pushforward_pca.svg`;
  - `conceptual_internal_cut.svg`;
  - `cnn_measured_controls.png`;
  - `cnn_measured_horizon.png`; and
  - `tracking_resolving.svg`.

The current post evidence is CNN-only. The one-model ResNet sweep, older CNN
depth-by-time sweep, VGG/criticality experiments, Part E, `report.html`, and the
delivered talk are not inputs to the post's quantitative claims.

Part E remains valuable recovery state and a measured negative diagnostic. Do
not delete or move `artifacts/part_e/gate_seed0_20260722/` or either
`diagnostic_v2_*` directory.

## Verified snapshot

- Full integrated CPU suite: **92 passed**, with one PyTorch AMP deprecation
  warning.
- The end-to-end LW fake-data smoke test passes using the repository `.venv`.
- Appendix SHA-256:
  `1c8d0ad028ddfdd083daf17e1703dd49b2cb426dfda761bd8003b59f914b2a95`.
- `cnn_measured_controls.png` SHA-256:
  `1cde8775af500597a9b2751fe727fd8a4c35f0a5dd2a3d0f21afdda7d920d8f5`.
- `cnn_measured_horizon.png` SHA-256:
  `b5d59e1b99b97cb28d9c05f3442f488af715ea10886531516bc393deb2c643f4`.
- The appendix is self-contained: no external scripts, styles, or image URLs.
- All local links in the relevant Markdown package resolve.
- The four headline contrasts recompute from the manifest inputs:

| Cut | Gaussian minus projected true | Mean-$r1$ minus Gaussian |
|---|---:|---:|
| After block 1 | $0.182913 \pm 0.055069$ | $0.144051 \pm 0.019072$ |
| After block 4 | $-0.012495 \pm 0.001530$ | $0.022739 \pm 0.010517$ |

The uncertainty above is sample standard deviation across three independently
trained CNN seeds. Each model has one surrogate draw; within-model redraw
variation was not measured.

The radius-zero seed-0 control requires a separate caveat. Its learning-rate
multiplier hit the `0.001` lower clip, and its first update norm was
`0.0007991132` versus `0.0007300291` for true replay: 9.463% larger, not exactly
matched. The centroid-versus-Gaussian endpoint reversal is therefore not yet a
clean exactly-matched comparison.

## Safe and unsafe claims

Safe:

- Under matched first-update norm, three epoch-30 CNNs show a large cut-1 versus
  cut-4 contrast after five replay epochs in the rank-2,048 retained subspace.
- The exploratory seed-0 rank-4,096 shallow cell passes the declared functional
  PCA gate and preserves the shallow ordering.
- One-model reinitialization and 20-epoch sensitivities preserve the qualitative
  early-versus-late contrast.

Unsafe:

- rank-2,048 full-space sufficiency at the shallow cut;
- “higher moments appear later”;
- a radius-independent conclusion that covariance beats means;
- an exact matched-update claim for the clipped radius-zero control;
- a representation-only depth effect, because receiver capacity and trainability
  change with the cut;
- optimizer equivalence beyond the first update;
- ResNet architecture-level replication; or
- Part E sequence-Gaussian sufficiency.

## Repository hazards to preserve

The current branch is 21 commits ahead of `main` and 19 commits ahead of its
remote tracking branch. Pages deploys only from `main`, while nothing under
`LW post/` is tracked yet.

The worktree also contains an unrelated restored ResNet/VGG regression and dirty
legacy checkpoint binaries. In particular, the dirty ResNet implementation
removes source lineage, dataset fingerprints, checkpoint hashes, projected-real
controls, nested PCA ranks, bank pairing, grid verification, and the tests that
enforced those properties. Its simplified tests pass only because the safeguards
were removed. Do not run it and do not include it in a publication commit.

Preserve both recovery surfaces until the work is integrated:

- `stash@{0}: pre-lw-rigorous-integration-2026-07-26`;
- `/home/crw/Documents/Vault/.worktrees/tracking2-lw`.

Never use `git add -A` in this repository. Before the ignore-rule repair, that
would have staged roughly 7.4 GB of local weights. Stage explicit paths and
inspect the complete cached file list and sizes before every commit.

Git still has about 5.05 GiB of reachable packs, much of it retained by
`refs/codex/turn-diffs/*`. Do not delete those refs, prune the stash/worktree, or
run an aggressive GC during active integration.

## Cleanup completed in this audit

- Removed the 2.0 GB incomplete duplicate at `artifacts/part_e/checkpoints/`
  after byte-comparing its 0M and 4M files with the verified canonical gate and
  confirming that its 16M model was only a truncated prefix. The complete gate
  remains recoverable at `artifacts/part_e/gate_seed0_20260722/`. The tombstone
  is `artifacts/part_e/deprecated/README.md`.
- Removed eight exact `tmp_pack_*` files that Git reported as garbage, reclaiming
  3.03 GiB. `git count-objects` now reports zero garbage and repository
  connectivity still passes.
- Moved the different-checkpoint rank gate and two superseded, unreferenced
  seed-0 JSONs under `artifacts/lw_post/deprecated/`.
- Expanded `.gitignore` so artifact `*.pt`, `*.safetensors`, artifact logs, and
  `papers/*.ps.gz` cannot be staged accidentally.
- Converted all 33 legacy inline-math delimiter pairs in `SPEC.md` to
  dollar-delimited Markdown math.
- Corrected stale Part E “planned/unrun” labels to measured one-seed negative
  diagnostic status.
- Corrected `README.md` to state the standing $10 RunPod cap and to warn that the
  named LW battery/verifier implement an obsolete protocol.
- Made `scripts/run_lw_post_smoke.sh` invoke the repository `.venv` explicitly,
  then removed repository Python/pytest caches, `artifacts/smoke/`, and the
  top-level generated `logs/` directory.

The repository shrank from about 22 GB to 17 GB. Experiment-local Part E and VGG
logs were preserved with their evidence directories.

## Provenance defect: repair metadata, not measurements

Three recorded full hashes are not Git objects. Their short prefixes resolve
unambiguously:

| Recorded value | Intended repository commit |
|---|---|
| `2078fea640cfd578586a833f72ad73ef1a012d28` | `2078fea64937ff7fdd89d11010d2d2ed24b7bfc8` |
| `5873b1bc8736e623effd57f83d976366458dedca` | `5873b1bfe329606d6848cdf01d40db3e4ac4f44a` |
| manifest `13aff88d1d4b09ec84f93e9bc3d37a47cd20962d` | `13aff88de2775df9073dda15c6174bcd5a7ed309` |

The fixed-LR cut-1 artifact uses the first invalid hash. The seed-0 training
manifest and embedded projection lineage use the second. The dashboard manifest
uses the third. Other publication artifacts using
`9b5acf99ad446c6e1dd026c7655e64d1a51f49e4` are exact.

The measured artifacts also record source-archive SHA-256 values, but those
archives are not retained locally. Do not silently rewrite raw result JSON. Add
an audited correction sidecar containing the recorded value, intended commit,
affected files/JSON pointers, unique-prefix evidence, and archive hash. Teach the
verifier/report to apply and disclose the correction. If an archive hash remains
the authoritative fallback, make the actual archive retrievable and verify its
digest. Update the manifest's own builder revision only after the final clean
layout/report commit exists.

## Work packages for later sub-agents

### Package 0 — source-control boundary and snapshot

Run this first, under one coordinator.

Goal: preserve the August 14 publication move and this audit's small cleanup
without mixing in the architecture regression or local weights.

Required actions:

1. inspect `git status`, `git diff`, the stash, and both worktrees;
2. stage only the publication package, its intended old-path deletions, the
   small deprecated JSON moves, documentation fixes, and relevant
   report/manifest/tests;
3. explicitly exclude dirty ResNet/VGG files, architecture runners/validators,
   modified checkpoint binaries, local raw artifact trees, and untracked helper
   scripts;
4. inspect every cached path and reject unexpected blobs before committing; and
5. create a recoverable snapshot commit on the current integration branch before
   any bulk archive move.

Acceptance criteria:

- no staged `*.pt`, `*.safetensors`, raw Part E checkpoint/data tree, or legacy
  architecture output;
- no staged changes to `src/tracking2/resnet_*`,
  `src/tracking2/vgg_suffix_statistics.py`, `tests/test_resnet_statistics.py`,
  `scripts/run_resnet_statistics.sh`, `scripts/run_vgg_suffix_statistics.sh`, or
  `scripts/validate_architecture_checkpoints.py`;
- the snapshot contains `LW post/` and the intended deletions from the old root,
  `docs/figures/`, and `notebooks/` locations; and
- the stash and secondary worktree remain intact.

### Package 1 — publication evidence contract and appendix

Own only:

- `artifacts/lw_post/dashboard_manifest.json` plus a new provenance sidecar;
- `scripts/make_lw_post_manifest.py`;
- `src/tracking2/post_report.py`;
- `tests/test_post_report.py`; and
- the generated canonical appendix.

Required actions:

1. Add an explicit manifest-level primary-analysis contract for epoch 30,
   cuts 1 and 4, rank 2,048, warm suffixes, matched first updates, five replay
   epochs, model seeds 0–2, and the two headline contrasts:
   Gaussian minus projected true and mean-$r1$ minus Gaussian.
2. Render that three-seed matched-update summary as the primary appendix result.
3. Relabel fixed LR as the optimizer-shock sensitivity; it must not remain the
   primary estimand.
4. Add the provenance-correction sidecar described above. Preserve raw JSON.
5. Make validation require resolvable Git commits or an explicit audited
   correction, rather than accepting any 40 hexadecimal characters.
6. Define a first-update matching tolerance and make clipped/floor-limited rows
   fail or display an explicit “approximately matched / clipped” status.
7. Regenerate the appendix and update tests around the corrected semantic
   contract.

Acceptance criteria:

- the appendix and post name the same primary regime and contrasts;
- the four headline values above are asserted from manifest inputs;
- fixed-LR and matched-update rows are never pooled;
- warm and reinitialized suffixes are never pooled;
- the projection gate still limits rank-2,048 claims to the retained subspace;
- the radius-zero row is not labelled exactly matched at its current 9.463%
  update-norm excess;
- invalid full hashes fail without the explicit correction sidecar; and
- a clean rebuild is deterministic.

### Package 2 — paid-run tooling and committed-artifact verification

Own the LW run/verify scripts only. Do not launch RunPod.

The current `scripts/run_lw_post_battery.sh`, `scripts/run_lw_post_gate.sh`, and
`scripts/verify_lw_post_artifacts.py` encode the obsolete rank-512,
all-checkpoint/all-cut, fixed-LR, three-draw, ten-epoch design. They do not
reproduce the July 26 publication battery.

Required actions:

1. Move or rename the old entry points into an explicit legacy/deprecated
   namespace, preserving their historical function.
2. Add a cheap verifier for the committed nine-input package, training
   manifests, correction sidecar, primary-analysis contract, headline values,
   figure inputs, and appendix hash/rebuild.
3. Either implement a resumable canonical publication runner exactly matching
   the manifest contract or leave a precise runner specification and state that
   no paid rerun is required. Do not leave a misleading executable with a
   current-sounding name.
4. Add a RunPod supervisor/termination trap and cost polling before any future
   paid use.

Acceptance criteria:

- invoking a current publication verifier checks the checked-in July 26 inputs;
- no current-sounding command launches the obsolete experiment;
- smoke outputs cannot pass as measured data; and
- no GPU or network access is used in this package.

### Package 3 — physical archive boundary

Run only after Package 0 creates a recoverable snapshot.

Goal: make the publication directory visually honest without deleting useful
scientific history.

Recommended layout:

```text
deprecated/2026-07-omnibus/
  README.md
  report.html
  artifacts/
    cifar_pilot_seed0/
    confirmatory/
    criticality/
    suffix_statistics/
    resnet_criticality/
    resnet_suffix_statistics/
    vgg_batchzoom/
    vgg_checkpoints/
    vgg_suffix_statistics/

LW post/deprecated/
  README.md
  talk/
  figures/
  notebooks/animate_training_dynamics.py
```

Keep only the five canonical figures listed above in `LW post/figures/`. Move
the talk and all of its referenced assets together so its `../figures/...`
links remain valid. Make `lw_post_figures.py` measured-only by default and route
legacy/full-mode outputs to the deprecated figure directory. Move root
`report.html` to the omnibus archive and either stop publishing it or update the
Pages copy path explicitly.

Update every affected README, generator input/output path, test, and Pages path.
Do not move Part E recovery state, papers, data, source code, tests, the manifest,
or its nine inputs.

Acceptance criteria:

- the current post package contains only current text, appendix, generator, and
  five referenced figures;
- every moved pilot is labelled in an archive README;
- the delivered talk is byte-identical and its relative assets resolve;
- measured-only figure generation does not recreate pilot files in the current
  directory; and
- the canonical appendix still rebuilds.

### Package 4 — editorial and final-link pass

Own `LW post/free-body-diagrams-for-neural-networks.md` only during the prose pass. Do not add new
substantive claims.

Required edits from the independent reader review:

1. Qualify the early mean-to-covariance “ladder” immediately: the experimental
   lowest rung is a radius-indexed mean-plus-isotropic-noise family, and only
   radius zero is a centroid-only condition. Use either $r$ or $\rho$
   consistently.
2. Put the native-receiver confound beside the main three-seed table/Figure 3,
   so a result-skimming reader cannot mistake the contrast for a
   representation-only depth effect.
3. State beside the main table that the three independent units are trained CNN
   seeds and that within-model surrogate-redraw uncertainty is unmeasured.
4. Correct the radius-zero wording: its LR multiplier hit the `0.001` floor and
   its first update was 9.463% above true replay. Call it clipped/approximately
   matched and soften the centroid inversion, or rerun that isolated cell with a
   lower/no floor before presenting it as exact matched evidence.

The publication release is final rather than provisional. Remove pre-release labels
consistently from the post, manifest, builder, appendix, README, and links.
Replace local-only relative appendix/image links
with tested final LessWrong or Pages asset URLs, or upload the assets through the
publication workflow.

Preserve the candid voice around “the result I am willing to keep is narrower”
and “stronger interpretations are unsafe.”

Acceptance criteria:

- a reader reaching Results already understands the nuisance radius;
- a reader skimming only the main result must mention the changing receiver;
- the replicated and unmeasured uncertainty units are explicit;
- no clipped row is described as exactly matched without a declared tolerance;
- every external literature claim is supported by the cited primary source;
- all final asset links work outside the local checkout; and
- final-release naming is internally consistent.

### Package 5 — independent verification and release integration

Run last, preferably in a clean worktree created from the intended publication
commit so the dirty architecture regression cannot mask omissions.

Required checks:

```bash
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=src

.venv/bin/python -m pytest -q -p no:cacheprovider
bash scripts/run_lw_post_smoke.sh
.venv/bin/python "LW post/notebooks/lw_post_figures.py" --measured-cnn-only
.venv/bin/python -m tracking2.post_report \
  artifacts/lw_post/dashboard_manifest.json \
  --output /tmp/tracking2-publication-appendix.html
cmp /tmp/tracking2-publication-appendix.html \
  "LW post/free-body-diagrams-for-neural-networks.html"
git diff --check
```

Also verify:

- manifest and correction-sidecar hashes;
- exact headline values and uncertainty units;
- zero forbidden Markdown math delimiters;
- zero missing local links and zero untested final-publication URLs;
- desktop and mobile appendix rendering;
- no unexpected figure or HTML diff after regeneration;
- no architecture/checkpoint regression in the publication commit; and
- Pages deployment after the intended commit is merged into `main`.

Do not call the work publication-ready until all checks pass on the exact commit
that will be deployed.

## Scheduling and file ownership

1. Package 0 is serial and comes first.
2. Packages 1 and 2 may run in parallel only if Package 2 does not edit the
   manifest/report files owned by Package 1. Have the coordinator make shared
   README changes afterward.
3. Package 3 begins after the snapshot and should not overlap with figure/report
   path edits.
4. Package 4 begins after Package 1 fixes the appendix contract, so prose and
   appendix language can be compared directly.
5. Package 5 is an independent final pass.

One agent owns each file. Agents should return concise diffs, tests, and blockers
rather than broad replacement drafts.

## Optional RunPod plan, only if a clean rerun is later chosen

The recommended spend remains $0. Historical manifest-listed runtime plus three
CNN trainings totals about 2.687 GPU-hours. If the author chooses an exact
clean-source replication after Packages 0–2:

- require at least 96 GB host RAM;
- accept only a pod at or below $1.50/hour;
- budget 3.5–4 hours and expect less than $6;
- hard-stop at the earlier of six hours or $9 accrued;
- run into a new staging directory from a clean, committed source;
- retain the actual source archive alongside its SHA-256;
- stop on source-revision failure, host-memory failure, hash mismatch,
  duplicate/missing cells, NaN/Inf, OOM, schema mismatch, or cost limit;
- download and verify locally; and
- terminate the pod immediately.

The user's hard cap is $10; ask before any action that could exceed it. The
current RunPod scripts lack an automatic budget cutoff and guaranteed
termination-on-error, so do not launch until that is repaired.

If the author wants to retain the strong centroid-versus-Gaussian inversion,
rerunning only the clipped radius-zero cell under an exact declared matching
tolerance is the one targeted measurement motivated by this audit. It is not
needed for the broader three-seed early-versus-late claim; otherwise qualify the
existing row instead.

ResNet/VGG reruns are separate Part C research and are not a publication
prerequisite. Part E should not be scaled to more seeds on the current negative
diagnostic.

## Prior-session context

The relevant earlier request was the August 14 Codex unification session, not
the stale Claude-web conversation index:

- prompt:
  `/home/crw/.codex/attachments/f19c2e35-e5b2-462f-a0a7-0c15ff35342e/pasted-text.txt`;
- session:
  `/home/crw/.codex/sessions/2026/08/14/rollout-2026-08-14T11-02-21-01a00170-7181-79f2-8d00-8768106cb59d.jsonl`.

That session completed the CNN-only narrative unification and deliberately
deferred GPU reruns, the dirty ResNet regression, and committing the directory
move. This handoff starts from that completed work rather than repeating it.
