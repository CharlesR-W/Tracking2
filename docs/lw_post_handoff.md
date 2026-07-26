# Handoff: Tracking2 LessWrong post

Date: 2026-07-26

## Current status

The user asked the previous agent to stop, revert the blog-post rewrite, and
leave a handoff for a different agent.

- `docs/lw_wip_post.md` has been restored exactly to commit `4371d3c`, the
  version from before this revision attempt.
- No new measured experiment completed. Do not describe the PCA/noise controls
  as results.
- All RunPod instances have been terminated. A final status query returned an
  empty pod list, so nothing should still be billing.
- Work happened in the isolated worktree
  `/home/crw/Documents/Vault/.worktrees/tracking2-lw`, on branch
  `codex/lw-post-revision`.
- The user's original checkout at
  `/home/crw/Programming/Experiments/Tracking2` was deliberately left
  untouched because it already contained substantial unrelated changes.

Important scope warning: only the prose file was restored in response to
"revert all changes to the blog post." Experimental code, dashboard work, and
new figure assets remain on the isolated branch for inspection. They should
not be treated as approved. Ask whether the user also wants the post-facing
figures/dashboard reverted before doing anything further.

## What was changed before the stop request

Three local commits were created:

1. `46ad955` — `Add reproducible internal-cut control battery`

   This adds an uninterrupted CNN checkpoint runner, PCA-projected-real
   controls, nested PCA-rank and mean-noise-radius sweeps, provenance and
   checkpoint-lineage recording, artifact verifiers, ResNet control code, and
   tests.

2. `0d0cf06` — `Rewrite LW note and publish measured-only appendix`

   This contained the now-reverted prose rewrite, redesigned figures/GIFs, a
   post-named measured-only dashboard, a reproduction note, and the exact
   legacy JSON files used by the dashboard.

3. `88db907` — `Support verified ResNet controls in data appendix`

   This lets the dashboard ingest the new nested schema-v3 ResNet artifact
   while retaining legacy schema support.

Before the prose restoration, the settled local suite reported 69 passing
tests. The fake-data end-to-end CNN smoke test passed. The dashboard rebuilt
byte-for-byte from its manifest, and the generated GIF metadata and selected
frames were inspected. These checks validate plumbing, not scientific claims.

## Most important findings

### Existing evidence is weaker than the original plots suggest

- The existing CNN points labelled by nominal epoch came from separately
  scheduled/trained models. They are useful as a depth pilot, but they are not
  checkpoints from one continuous trajectory and should not be used to infer
  change over training.
- The legacy ResNet artifacts do come from one training trajectory, but they
  lack the source/checkpoint lineage of the new protocol. They also re-encoded
  stochastic crop/flip augmentations at each cut, so depth is partly confounded
  with the realized input bank.
- Repeated surrogate draws vary generated activations and minibatch order.
  They are not independent trained-model seeds.

### Mean-plus-noise needs a precise definition

The implemented default is not optimizer epsilon and not the tiny Cholesky
jitter. At radius 1, the isotropic covariance trace equals the pooled average
within-class covariance trace in retained PCA space. Radius 0 gives exact class
centroids; covariance trace scales as radius squared. The proposed CNN sweep is
`0, 0.1, 1, 2`. The numerical Cholesky jitter is separately `1e-6 I`.

This ablation is methodologically useful, but it has not been run.

### PCA needs controls rather than "all components"

At shallow cuts, native activations have tens of thousands of coordinates.
Full dense per-class covariance is sample-rank limited and expensive to
factorize. The implemented alternative is:

- one maximal PCA basis per cut;
- nested leading-coordinate ranks;
- class moments estimated from the full training activation bank after the
  basis is fixed;
- held-out total, within-class, and between-class coverage;
- a PCA-projected-real condition to separate discarded-direction loss from the
  Gaussian approximation.

The intended CNN ranks are `128, 512, 1024`; the ResNet control uses
`128, 256, 512`.

### The main unresolved confounds remain

- Moving the cut changes suffix capacity.
- Warm-started suffix training mixes retention, relearning, and forgetting.
- A fixed relaxation budget mixes adaptation speed with eventual capability.
- Synthetic samples may fall off the normal activation manifold even when
  moments match.
- Independent model seeds and a capacity-matched receiver are still missing.

These are more important than polishing the prose into a paper-like claim.

## RunPod attempt and why it stopped

The approved source-only archive was:

- path: `/tmp/tracking2-lw-source.tgz`
- source commit: `46ad955e6b49695f4ba0b629685c90210cc063f4`
- SHA-256:
  `c67f502375ea24e763d5623183469e0cd77b1f55cfa8fb8a36e315429649ca84`

No measured battery was launched.

Two secure A40 pods advertised large host RAM but exposed only a 50 GB cgroup
limit. The 90 GiB protocol guard rejected both, and both were terminated.

A third request used RunPod's `minRAMPerGPU=96` option and obtained:

- 125 GB cgroup allocation (116.4 GiB visible);
- 16 vCPUs;
- RTX 4090;
- `$0.69/hour`.

That pod passed the RAM and CUDA checks. It was also terminated when the user
asked the agent to stop.

Environment lesson: the base image had PyTorch 2.1/CUDA 11.8. Running a full
`uv sync` would replace that stack with much newer CUDA packages, so a
system-site-packages venv was used instead. The battery dependencies imported
successfully. Installing the lock's `transformers==4.57.6` then made full test
collection incompatible with PyTorch 2.1:

`torch.utils._pytree` lacked `register_pytree_node`.

This occurred before the smoke test or measured run. A future agent should
either use a compatible Transformers version for the PyTorch 2.1 image, choose
a newer base image matching the lock, or run only the battery-relevant tests
after documenting that choice. Do not silently skip the incompatibility.

## Post/dashboard/figure state

Useful files left for inspection:

- `docs/lw_wip_post.md` — restored original post.
- `docs/lw_post_reproduction.md` — proposed bounded protocol.
- `free-body-diagrams-for-neural-networks.html` — measured-only dashboard built
  from hashed legacy evidence.
- `artifacts/lw_post/dashboard_manifest.json` — exact dashboard inputs.
- `notebooks/lw_post_figures.py` — redesigned figure/GIF source.
- `scripts/run_lw_post_battery.sh` — proposed uninterrupted CNN battery.
- `scripts/verify_lw_post_artifacts.py` — evidence gate.

One known integration trap: `cnn_endpoint_bands_figure()` in
`notebooks/lw_post_figures.py` is still hard-wired to legacy CNN artifacts.
If new CNN results are ever generated, that panel must be changed or removed
before claiming the figures use the uninterrupted trajectory.

The slideshow containing the middle-out framing was found at:

`/home/crw/Programming/Experiments/Tracking2/Tracking2-talk-slides.html`

There are many unrelated or copied untracked artifacts in the isolated
worktree. They were intentionally not deleted or committed.

## Questions for the user before another agent proceeds

1. Does "revert all changes to the blog post" mean only the Markdown prose, as
   assumed here, or also every new figure, dashboard, reproduction note, and
   post-facing commit?
2. Should the next pass be editorial only, with no experiment engineering?
3. Which two or three figure changes matter most? It would be safer to agree on
   a storyboard before regenerating the full suite.
4. How prominent should the middle-out/alignment framing be relative to the
   narrow experimental result?
5. What evidence threshold is desired for this WIP: cleaned pilot disclosure,
   the CNN controls, independent model seeds, or something else?
6. Should commit `46ad955` be kept as useful future experiment infrastructure
   or reverted with the presentation work?

## Recommended next-agent approach

Start by reading the restored post and the user's original request, then ask
the scope questions above. Make a small, reviewable editorial diff first.
Avoid another full rewrite, dashboard rebuild, or GPU run until the user has
seen and approved that direction.

If experiments are requested again, begin with a battery-relevant smoke test
on a compatible environment, then run one bounded artifact and inspect it
before launching the full grid. Never put an unrun control into the public
evidence surface.
