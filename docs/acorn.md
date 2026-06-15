# ACORN - Adjusted CORrelations, Negative-binomial

ACORN is a hierarchical Negative-Binomial model that tests, for *every* feature in a count
table simultaneously, whether its abundance is associated with a continuous marker (a
blood/clinical measurement, a phenotype score, etc.), while adjusting for a set of covariates —
all in a single joint fit. It was developed for microbiome genus × blood-marker associations
but applies to any over-dispersed count table with a continuous marker of interest.

## Why one joint model instead of one GLM per feature?

Running an independent GLM per feature and then applying an FDR correction is the standard
approach, but it treats each feature's estimate independently and then bluntly thresholds the
result. ACORN instead places a **hierarchical (partial-pooling) prior** on every per-feature
coefficient: each feature's intercept and slopes are drawn from a shared population distribution
(`coef[feature] = mu + tau * z[feature]`). Features with little data or noisy signal are
automatically pulled toward the population mean ("shrinkage"), while features with strong,
consistent signal keep their own estimate. This is a principled, data-driven alternative to
ad-hoc multiple-testing correction, and it gives a full posterior per feature instead of a
point estimate + p-value.

## Mathematical formulation

The data is reshaped to long format: one row per `(sample, feature)` pair. For row $i$
belonging to feature $t = \text{feature\_idx}[i]$:

$$
y_i \sim \mathrm{NegativeBinomial}\!\left(\mu = e^{\eta_i},\; \alpha = \phi_t\right)
$$

$$
\eta_i = \beta^{0}_t + \beta^{\text{marker}}_t \, m_i
       + \sum_{c} \beta^{c}_t \, x^{c}_i
       + o_i
$$

where $m_i$ is the (z-scored) marker, $x^c_i$ are the covariates and $o_i$ is an optional
$\log(\text{total})$ offset (off by default; intended for count tables that are *not* normalized
to even sequencing depth).

Every per-feature coefficient $X \in \{\beta^0, \beta^{\text{marker}}, \beta^c\}$ uses a
**non-centered** hierarchical prior:

$$
\mu_X \sim \mathcal{N}(0, 1), \quad
\tau_X \sim \mathrm{HalfNormal}(1), \quad
z_X[t] \sim \mathcal{N}(0, 1), \quad
X[t] = \mu_X + \tau_X \, z_X[t].
$$

Non-centering ($\mu + \tau z$ instead of $\mathcal{N}(\mu, \tau)$ directly) keeps NUTS
well-behaved when $\tau_X$ is near zero — the common situation here, since most features have
*no* association with most markers.

The per-feature dispersion $\phi_t$ is by default independent, $\phi_t \sim \mathrm{HalfNormal}(1)$
(rare features tend to be noisier than abundant ones). With `hierarchical_dispersion=True` it is
instead partially pooled on the log scale, $\phi_t = \exp(\mu_\phi + \tau_\phi z_\phi[t])$.

The NB uses the mean/dispersion parametrization $\mathrm{Var}(y) = \mu + \mu^2/\alpha$ — larger
$\alpha$ approaches Poisson, smaller $\alpha$ is more overdispersed.

## Usage

```python
import pandas as pd
from prodiphy import ACORN

# Wide table: one row per sample, with feature count columns + marker + covariates.
df = pd.read_csv("samples.csv")
feature_cols = [c for c in df.columns if c.startswith("g_")]

model = ACORN()
model.fit(
    df,
    count_cols=feature_cols,
    marker="CRP_mgL",
    covariates=[
        ("age", "continuous"),       # z-scored
        ("BMI", "continuous"),       # z-scored
        ("gender", {"m": 1, "f": 0}),  # explicit 0/1 binary contrast
    ],
    method="nuts",                   # "advi"/"fullrank_advi" for a fast preview
)

table = model.get_stats()            # ranked association table
```

Continuous covariates (and the marker) are z-scored before fitting, so all slopes are on a
"per 1 SD" scale and the `Normal(0, 1)` / `HalfNormal(1)` priors are weakly informative
regardless of the original units. Binary covariates require an **explicit** `{value: 0/1}`
mapping — the sign of the coefficient depends on which level maps to `0` vs `1`, so it is never
inferred. Multi-level (k>2) categorical covariates are rejected; encode them as separate binary
covariates instead.

### Inference

Use `method="nuts"` for any result that will be reported. NUTS runs through PyMC's default
sampler (`nuts_sampler="pymc"`); pass `nuts_sampler="numpyro"` for GPU-accelerated sampling if
`jax`/`numpyro` are installed. `method="advi"` / `"fullrank_advi"` are fast variational previews
for model-development iteration only — they systematically underestimate uncertainty in
hierarchical models, so do not take their HDIs to a manuscript.

## Output

`get_stats(rope_log=0.05, hdi_prob=0.94)` returns one row per feature, ranked by strength of
evidence:

| column | meaning |
|---|---|
| `feature` | the feature (count column) name |
| `log_effect` | posterior median slope, on the log-abundance scale, per 1 SD increase in the marker, adjusted for covariates |
| `hdi_94_low` / `hdi_94_high` | highest-density interval of the posterior at the requested probability |
| `fold_change` | `exp(log_effect)`, the multiplicative change in abundance per 1 SD of the marker |
| `prob_direction` | posterior mass on the side of zero where most of the posterior lies (0.5 = no evidence of direction, 1.0 = certain). The Bayesian analog of "this is non-zero" |
| `prob_outside_rope` | fraction of the posterior outside a "region of practical equivalence" around zero (`rope_log`, default 0.05 on the log scale ≈ a 5% abundance change). High = not just non-zero, but large enough to matter |
| `hdi_excludes_zero` | the simple, conservative "is this credible" flag — `True` when the HDI doesn't straddle zero |

`get_summary(var_names=...)` returns a raw ArviZ posterior summary (means, HDIs and sampler
diagnostics); the group-level `mu_*` / `tau_*` parameters are useful for a quick "is there *any*
population-level signal" check before drilling into per-feature results.
