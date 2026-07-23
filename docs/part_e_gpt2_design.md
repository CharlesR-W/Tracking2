# Part E design: Part B suffix relaxation for next-token prediction

> **Status: PLANNED / UNRUN.** No Part E model or suffix-relaxation artifact
> exists. Dashboard panels remain marked **MOCKUP — PLANNED / UNRUN** until a
> validated measured artifact replaces them.

## 1. Question

Part E asks one primary question:

> At a frozen GPT-2 residual-stream interface, are a next-token-conditioned mean
> and modeled second-order sequence statistics sufficient for the downstream
> suffix under the same finite relaxation protocol used in Part B?

At checkpoint $t$ and cut $\ell$, write

$$
H=\phi_{t,\ell}(x), \qquad f_t(x)=\psi_{t,\ell}(H),
$$

where $H\in\mathbb R^{T\times d}$ is a complete residual-stream sequence.
Part E freezes $\phi$, fits three distributions over $(H,Y)$, warm-starts three
identical suffixes from $\psi_{t,\ell}$, relaxes each on one distribution, and
cross-evaluates every suffix on every held-out distribution.

This is the Part B experiment with the smallest changes needed for causal
next-token prediction. Instruction tuning, LoRA, RL, moment velocity,
Fisher/GGN, reset, and bandwidth are not part of the first Part E result.

## 2. B-to-E mapping

| Part B | Part E | Reason for the change |
|---|---|---|
| activation vector $z$ | complete sequence $H$ plus mask and positions | the suffix is causal and order-sensitive |
| class label $y$ | next-token target $y_p=x_{p+1}$ | defines the supervised joint distribution at each loss-bearing position |
| class-conditional Gaussian | sequence Gaussian with channel and cross-position covariance | IID token rows delete the structure being tested |
| held-out accuracy | next-token cross-entropy in nats per loss-bearing token | primary NTP loss; lower is better |
| example bootstrap | story-level bootstrap | tokens within a story are dependent |
| ordinary suffix copy | suffix blocks, final LayerNorm, and cloned untied LM head | suffix updates must not alter the frozen input embedding through weight tying |

Everything else follows Part B: identical warm starts, labels, sample counts,
minibatch order, optimizer, learning rate, update count, and held-out banks.

## 3. Bounded first experiment

Use GPT-2 small (12 blocks, 12 heads, width 768) at context length 256 on a
deterministic GPT-2-tokenized TinyStories subset. The scratch-training trajectory
is the only required training arm.

- Save selected early, middle, and final checkpoints on an absolute loss-token axis.
- Cache `resid_post` after blocks 0, 5, and 11 for the first gate.
- Use fixed story-level fit, relaxation, and held-out banks.
- Store token IDs, targets, attention masks, position IDs, loss masks, checkpoint
  hashes, and replay hashes.
- Disable dropout during replay, or pair its randomness exactly across conditions.

The first gate uses one model seed, one independently keyed surrogate draw, and
64 suffix updates. Confirmatory work uses three model seeds, three surrogate draws
per seed, and 128 updates only after all mechanical checks and at least one
preregistered true-row contrast pass.

## 4. Three activation distributions

Fit a local PCA projection at each checkpoint and cut. Let
$Z=(H-\bar H)U$ be the retained coordinates. The PCA is an operational
approximation, not a cross-checkpoint motion basis.

### 4.1 Mean-only

Use an additive conditional mean

$$
M_p(Y)=\bar z+a_{g(y_p)}+b_{\operatorname{posbin}(p)},
$$

where frequent next-token types receive singleton groups and rare tokens use
preregistered frequency/token-shape groups. Add calibrated isotropic nuisance in
the retained subspace. This is the direct analogue of Part B's class-mean proxy.

Conditioning on $y_p$ is legitimate for this joint-distribution diagnostic: the
label is present during suffix relaxation just as a class label is present in
Part B. It does not define an inference-time generator. A history-only mean is a
secondary ablation.

### 4.2 Sequence Gaussian

Use the same mean and a bounded matrix-normal proxy,

$$
Z\mid Y\sim\mathcal{MN}\!\left(M(Y),K_{\mathrm{pos}},
\Sigma_{\mathrm{chan}}\right).
$$

$\Sigma_{\mathrm{chan}}$ models covariance across retained channels and
$K_{\mathrm{pos}}$ models covariance across valid within-story positions. Fix the
matrix-normal scale, declare shrinkage, and validate held-out lag covariances.
This separable approximation is the Part E meaning of “Gaussian”; the claim must
not silently expand to arbitrary sequence covariance.

### 4.3 True

Replay the full cached residual-stream sequence with its original mask, positions,
and targets. This is the full-space reference.

## 5. Relax-by-evaluate matrix

For $r,s\in\{\mathrm{mean},\mathrm{seqG},\mathrm{true}\}$, record

$$
M^{(t,\ell)}_{s,r}(u)=
\mathbb E_{(H,Y)\sim Q_s}
\ell_{\mathrm{NTP}}\!\left(\psi^{(u)}_{t,\ell;r}(H),Y\right).
$$

The full $3\times3$ matrix is required. The primary statistic is true-activation
excess loss,

$$
\Delta_{\mathrm{true}\mid r}^{(t,\ell)}(u)=
M_{\mathrm{true},r}^{(t,\ell)}(u)-
M_{\mathrm{true},\mathrm{true}}^{(t,\ell)}(u).
$$

Near-zero sequence-Gaussian excess loss with materially larger mean-only excess
loss supports the bounded claim that the modeled second-order sequence statistics
are sufficient for this checkpoint, cut, PCA proxy, and relaxation budget. A
positive sequence-Gaussian gap says the fitted proxy misses useful structure; it
does not alone identify that structure as higher-order.

## 6. Necessary validity checks

These checks gate interpretation but do not become separate headline panels.

1. **Step-zero replay parity:** the cached suffix reproduces intact-model logits
   and cross-entropy before any update.
2. **Tied-head isolation:** clone and detach the LM-head weight for each suffix
   copy, then repeat the step-zero parity test.
3. **Immutable banks:** fit, relaxation, and held-out story sets are disjoint and
   hash-stable across every condition.
4. **Causal leakage guard:** changing targets after position $p$ must not change
   generated rows at or before $p$.
5. **Surrogate fit:** validate conditional means, channel covariance, lag
   covariance, support/range, and common/tail-token strata on held-out stories.
6. **PCA guard:** report held-out variance coverage and compare projected-true
   replay with full-true replay. A material functional gap blocks a full-space
   sufficiency claim.
7. **Independent uncertainty:** model seeds support training-level claims;
   surrogate draws quantify measurement variation; tokens and checkpoints are
   not independent replicates.

An IID-token Gaussian is a useful secondary ablation for the contribution of
cross-position covariance. Projected-true replay is a diagnostic for discarded
coordinates. Neither expands the headline matrix beyond $3\times3$.

## 7. Artifacts

Persist:

- `manifest.json`: code, model, tokenizer, data, checkpoint, dtype, and config hashes;
- `replay_manifest.json`: story IDs, tokenization, positions, masks, and split hashes;
- `sequence_surrogates.npz`: PCA basis, conditional mean terms, covariance factors,
  coverage, and fit seed;
- `surrogate_diagnostics.parquet`: held-out fit, leakage, and projected-true checks;
- `suffix_statistics.json`: all nine relaxation/evaluation paths, step-zero parity,
  suffix provenance, and failure flags.

The dashboard must rebuild from these saved artifacts without importing training
code. Failed or missing cells remain visible as missing evidence.

## 8. Deferred extensions

After the scratch-training replication is valid, the same protocol can be reused
for ordinary continuation, instruction full-token LM, response-only SFT, LoRA, or
preference/RL training. Those comparisons need their own matched exposure and
causal claims; they are separate experiments rather than prerequisites for Part E.

Finite resolving/transfer demand may remain an audit diagnostic. Part D retains
ownership of moment-speed normalization, directional Fisher/GGN, causal
transports, reset, and frequency-response claims.
