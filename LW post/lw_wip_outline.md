# Working outline: free-body diagrams for neural networks

## Publication scope

The post's empirical center is the July 26 CNN control battery. It is a bounded
report about what frozen suffixes can learn from selected statistics of their
internal activation distributions.

- **Canonical evidence:** three independently trained four-block CNNs at epoch
  30, cuts after blocks 1 and 4, rank 2,048, matched first-update norms, and five
  relaxation epochs.
- **Primary metric:** held-out true-activation cross-entropy. Accuracy is a
  secondary, intuitive readout.
- **Canonical figures:** `figures/cnn_measured_controls.png` and
  `figures/cnn_measured_horizon.png`.
- **Historical only:** older CNN depth-by-time and one-model ResNet figures. They
  may appear only in a clearly labelled pilot/methods section.
- **Separate negative extension:** Part E is mechanically valid but does not
  carry the CIFAR argument.
- **Out of scope:** Part D, longitudinal tracking claims, and architecture-level
  replication.

Use three labels throughout:

- **Observed:** directly supported by the declared finite protocol.
- **Interpretation:** a possible explanation, not identified by the experiment.
- **Next test:** a measurement that could distinguish explanations.

## One-sentence claim

> Under matched first-update norms, three independently trained CNNs show a
> robust contrast between early and late cuts in the rank-2,048 retained
> subspace: after five relaxation epochs, shallow suffixes distinguish projected
> real, exact-Gaussian, and mean-$r1$ replay much more strongly than the late
> native suffix does.

Do not shorten this to “higher moments appear later.” The shallow rank-2,048
predictive-KL PCA gate fails, cut depth changes receiver capacity, and the
Gaussian-versus-mean comparison depends on mean-noise radius.

## Reader route

### 0. Short version

Open with the result and its boundaries:

1. Every prefix induces an effective dataset for the suffix.
2. Freeze the prefix, fit selected-statistics replay distributions, relax cloned
   suffixes, and evaluate them on held-out real activations.
3. The early/late contrast replicates across three trained CNN seeds after
   matching the first update.
4. The result is a projected-subspace, finite-horizon receiver comparison, not a
   located onset of cumulant order.

The drafting disclosure can remain:

> The experiments and scientific judgments are mine; drafting and organization
> were assisted by AI writing/coding tools.

### 1. From input statistics to internal effective datasets

Give only the background needed for the question. A Gaussian is the
maximum-entropy distribution fixed by mean and covariance; distributional
simplicity work asks when networks use statistics beyond those constraints.
Then make the middle-out move:

$$
x \xrightarrow{\phi_{t,\ell}} z
\xrightarrow{\psi_{t,\ell}} \hat y,
\qquad
P_{t,\ell}=\operatorname{Law}(z,y).
$$

Treat $P_{t,\ell}$ as the suffix's effective dataset. Ask which fitted structure
supports further suffix learning at a fixed checkpoint and cut. Do not imply
that this frozen-interface experiment measures how the interface moves during
ordinary end-to-end training.

### 2. The controlled replay experiment

Use the protocol schematic and explain the four relevant distributions:

1. full true activations;
2. true activations projected into and reconstructed from the PCA subspace;
3. an exact empirical class Gaussian in that subspace; and
4. class means with declared isotropic nuisance radius.

Clone the checkpoint suffix from an identical warm start, relax each copy, and
evaluate on held-out true activations. Matching rescales each condition to the
true condition's first-update norm; it does not match later optimizer dynamics.

The clean contrasts are:

- full true versus projected true: projection damage;
- projected true versus Gaussian: within-subspace non-Gaussian mismatch; and
- Gaussian versus mean-$r$: covariance-plus-noise comparison at the declared
  radius.

### 3. Adequacy before interpretation

Put the PCA table before the main result.

| Rank | Total variance | Within-class variance | Projected − true CE | Predictive KL | Gate |
|---:|---:|---:|---:|---:|:---:|
| 2,048 | 0.8802 | 0.8760 | +0.0154 | 0.0373 | fail |
| 3,072 | 0.9121 | 0.9090 | +0.0083 | 0.0211 | fail |
| 4,096 | 0.9303 | 0.9278 | +0.0045 | 0.0146 | pass |

The gate is projected-minus-true update-0 CE at most 0.05 nat and predictive KL
at most 0.02 nat. Therefore the three-seed rank-2,048 shallow comparison is a
projected-subspace estimand. The rank-4,096 pass is exploratory, seed 0 only,
and selected after inspecting the held-out test population.

### 4. Main result: replicated early/late contrast

Lead with `figures/cnn_measured_controls.png`. Report endpoint held-out CE after
five epochs:

| Cut | Gaussian − projected, mean ± SD | Mean-$r1$ − Gaussian, mean ± SD |
|---|---:|---:|
| after block 1 | +0.183 ± 0.055 | +0.144 ± 0.019 |
| after block 4 | −0.012 ± 0.002 | +0.023 ± 0.011 |

The three independent seeds all show the same qualitative contrast. At the
shallow cut, Gaussian and mean-$r1$ replay lag projected-real replay; at the
late cut, the native suffix is comparatively insensitive under this finite
protocol.

Immediately state what this does not show: full-space shallow sufficiency, a
specific higher-order mechanism, an onset time, or capacity-matched receivers.

### 5. Controls that changed the claim

Keep these close to the main figure rather than in a defensive appendix.

#### Optimizer scale

With fixed LR at seed 0/cut 1/rank 2,048, first-update norms were approximately
2.5x projected-real, 184x Gaussian, and 287x mean-$r1$ relative to true replay.
Matching the first update sharply reduces the endpoint gaps but preserves their
ordering. Most of the fixed-LR magnitude was an optimizer-scale artifact.

#### Mean-noise radius and covariance estimator

At seed 0/cut 1, exact centroids beat the Gaussian while radius-1 mean replay
loses to it; radius 2 is again approximately tied with the Gaussian. Therefore
“covariance helps” is not radius-robust. Five-percent covariance shrinkage is a
small, separate sensitivity and does not drive the seed-0 result.

#### Adequacy-passing exploratory cell

At seed 0/cut 1/rank 4,096, endpoint CE is 0.938 projected-real, 1.065 exact
Gaussian, and 1.233 mean-$r1$. The shallow ordering survives where the declared
PCA gate passes, but this one test-selected cell is not confirmatory.

#### Warm start, reinitialization, and horizon

Reinitializing suffixes preserves the qualitative early/late contrast, but the
fresh shallow true suffix is itself much weaker after five epochs. This
strengthens the receiver-capacity warning. Extending the warm suffix to 20 epochs
does not close the shallow gap; `figures/cnn_measured_horizon.png` shows this
control. Neither intervention capacity-matches cuts.

### 6. Interpretation: a free-body diagram, not a mechanism claim

Frame the result as forces on an internal learner:

- the statistical structure available at its interface;
- the receiver's inherited state and trainable capacity;
- optimizer scale and finite relaxation horizon; and
- projection and surrogate-model error.

The experiment isolates some of these forces but not all. It supports a robust
early/late operational contrast. It does not establish that higher moments are
created, destroyed, or acquired at a particular depth.

### 7. Historical pilots and architecture boundary

If the older animations help explain the method, place them in a section titled
**Earlier pilot / methods demonstration**. State that:

- the older CNN sweep spans epochs and cuts not covered by the July 26 battery;
- the ResNet trajectory uses one trained model and has incomplete lineage; and
- neither figure supports the post's canonical quantitative claims or an
  architecture-level replication.

The talk under `talk/` is frozen as delivered and uses these preliminary pilots.
Do not update it silently or use it as a second evidence surface.

### 8. Negative sequence-model extension

Part E can receive one short, separate note. The repaired one-seed 50M-token
diagnostic passes adaptive-PCA, projected-replay, fp32-parity, and leakage gates.
After gradient matching, sequence Gaussian does not beat mean replay at cuts 0
or 5 and only modestly improves at cut 11. This is a useful negative diagnostic,
not support for the CIFAR claim; redesign should precede multi-seed work.

### 9. Safe conclusion and next tests

End on the observed contrast and the experiments that could sharpen it:

- capacity-matched receivers across cuts;
- independent seeds and fresh rank selection for an adequacy-passing shallow
  comparison;
- optimizer controls beyond the first update;
- a less arbitrary mean-only nuisance distribution; and
- genuine depth-by-time evidence before making representation-motion claims.

The canonical appendix is the CNN-only
`free-body-diagrams-for-neural-networks.html`, rebuilt from
`../artifacts/lw_post/dashboard_manifest.json`. `../report.html` is an omnibus
research archive, not the post's evidence index.
