# Handoff: Genus × Blood-Marker Hierarchical Negative-Binomial Correlation Model

**Source:** `notebooks/Final - Supplemental Table S12b - genera health pymc.ipynb`
(Healthy Gut Project — "Delineation and prognostic potential of the healthy gut
microbiome", Supplemental Table S12b)

**Goal of this document:** describe a PyMC model that tests, for *every* microbial
genus simultaneously, whether its abundance is associated with a continuous blood
marker (CRP, HbA1c, fasting insulin, calprotectin, etc.), adjusted for age, BMI and
sex — and provide everything needed to lift it into a shared PyMC-model package in a
more general form. The age/BMI/sex handling is intentionally called out as the main
thing that needs a more general design; everything else is closer to "drop-in".

---

## 1. Scientific context / what the model answers

For a cohort of `P` participants we have:

- A **genus-level count table**: for each participant, the read counts of `G`
  microbial genera from 16S rRNA gene amplicon sequencing, **rarefied to an even
  sequencing depth** (`data/EvenSampleDepth_genus_matrix.tsv`). Because depth is
  already equalized across samples, raw counts are comparable between participants
  *without* a library-size offset term.
- One **marker of interest** per participant — a continuous clinical/blood variable
  (e.g. `CRP_mgL`, `HbA1c.`, `Insuline_nuchter_mUL`, `calpro_final.ug.per.g`,
  `health_penalty`, ...).
- A small set of **covariates** to adjust for: `age`, `BMI`, `sex`.

The question: *"adjusting for age/BMI/sex, is this genus's abundance associated with
this marker — and how does that association compare across ~200+ genera at once?"*

Rather than running ~200 independent GLMs and correcting for multiple comparisons
(FDR), the model fits **all genera jointly in one hierarchical model**. Each
genus gets its own intercept and slopes, but those per-genus parameters are
*partially pooled* toward a shared group distribution. This:

- automatically shrinks noisy/rare-genus estimates toward the population mean
  (a principled alternative to FDR correction),
- lets well-measured genera retain their own signal,
- yields a full posterior per genus, so "is this real?" is answered with
  credible intervals / probability-of-direction rather than p-values.

The output is one ranked table per marker: for every genus, the adjusted
log-effect of the marker on its abundance, a 94% HDI, a fold-change, and two
posterior-probability summaries (`prob_direction`, `prob_outside_rope`).

---

## 2. Statistical model

Data is reshaped to **long format**: one row per `(participant, genus)` pair,
`N = P * G` rows total.

For row `i` belonging to genus `g = genus_idx[i]`:

```
y_i ~ NegativeBinomial(mu = exp(log_mu_i), alpha = dispersion[g])

log_mu_i = intercept[g]
         + beta_marker[g] * marker_i
         + beta_age[g]    * age_i
         + beta_bmi[g]    * bmi_i
         + beta_sex[g]    * sex_i
```

Every per-genus coefficient (`intercept`, `beta_marker`, `beta_age`, `beta_bmi`,
`beta_sex`) is **partially pooled** across the `G` genera via a non-centered
hierarchical prior:

```
mu_X    ~ Normal(0, 1)                  # group-level mean of coefficient X
tau_X   ~ HalfNormal(1)                 # group-level spread
z_X[g]  ~ Normal(0, 1)                  # genus-specific offset (non-centered)
X[g]    = mu_X + tau_X * z_X[g]         # deterministic, per-genus coefficient
```

Non-centering (`mu + tau * z` instead of `Normal(mu, tau)` directly) keeps NUTS
well-behaved when `tau_X` is near zero — a common situation here, since most
genera have *no* association with most markers.

`dispersion[g] ~ HalfNormal(1)` is **not** hierarchically pooled — each genus gets
an independent overdispersion parameter (rare genera tend to be much noisier than
abundant ones, and the per-genus prior already captures most of the needed
heterogeneity).

`marker`, `age`, `bmi` are z-scored (mean 0, sd 1) before fitting, so all slopes
are "effect per 1 SD" and priors are on a sensible scale. `sex` is coded as 0/1.

The `beta_marker[g]` coefficients are the quantity of interest; `beta_age`,
`beta_bmi`, `beta_sex` are nuisance/adjustment terms carried for completeness
(mirroring the covariate adjustment of the original frequentist GLM this model
replaces).

---

## 3. Code

All code below is taken verbatim from the notebook (only the `# hba1c_idata = ...`
style "scratch" lines are omitted). It assumes `pandas`, `numpy`, `pymc`, `arviz`.

### 3.1 Helpers: standardizing covariates and encoding sex

```python
# Standardizing continuous predictors keeps the priors on a sensible scale
# and makes slopes comparable across genera. We return the mean/std so the
# marker can be back-transformed later if needed for reporting.
def standardize(series):
    values = series.to_numpy(dtype=float)
    mean, std = np.nanmean(values), np.nanstd(values)
    return (values - mean) / std, mean, std


# Sex may arrive as strings ("M"/"F"), booleans, or already-numeric. We map it
# to a clean 0/1 contrast so the slope reads as "effect of the reference level".
def encode_sex(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=float)
    categories = series.astype("category")
    # 0/1 against the first category alphabetically; note which level is the reference.
    return categories.cat.codes.to_numpy(dtype=float)
```

### 3.2 `build_model_data`: wide participant table -> long model-ready arrays

```python
def build_model_data(
    merged_df, genera, marker_name, covariates=("age", "BMI", "gender")
):
    age_col, bmi_col, sex_col = covariates

    # Drop participants missing the marker or any covariate up front, so every
    # genus row for a given participant is built from complete predictor data.
    needed = [marker_name, age_col, bmi_col, sex_col]
    df = merged_df.dropna(subset=needed).reset_index(drop=True)

    n_participants = len(df)
    genus_names = list(genera)
    n_genera = len(genus_names)

    # Wide -> long: each participant contributes one row per genus.
    # np.tile repeats the per-participant covariate across all its genus rows;
    # the genus block is the fastest-varying axis so genus_idx tiles cleanly.
    abundances = df[genus_names].to_numpy(dtype=float)  # shape (P, G)
    y = abundances.reshape(-1)  # row-major: participant-major

    genus_idx = np.tile(
        np.arange(n_genera), n_participants
    )  # 0..G-1 repeated per participant

    marker_std, _, _ = standardize(df[marker_name])
    age_std, _, _ = standardize(df[age_col])
    bmi_std, _, _ = standardize(df[bmi_col])
    sex_code = encode_sex(df[sex_col])

    # Broadcast each per-participant covariate to the long layout.
    repeat_per_participant = lambda v: np.repeat(v, n_genera)

    data = {
        "y": y,
        "genus_idx": genus_idx,
        "marker": repeat_per_participant(marker_std),
        "age": repeat_per_participant(age_std),
        "bmi": repeat_per_participant(bmi_std),
        "sex": repeat_per_participant(sex_code),
        "genus_names": genus_names,
    }
    return data
```

### 3.3 `fit_genus_marker_model`: the PyMC model + fitting

```python
import pymc as pm
import numpy as np


def fit_genus_marker_model(
    model_data,
    method="nuts",
    target_accept=0.9,
    random_seed=42,
    nuts_sampler="numpyro",
    draws=500,
    tune=500,
    advi_iterations=30000,
    advi_draws=1000,
):
    """Fit the hierarchical NB model relating one blood marker to all genera.

    model_data is the dict returned by build_model_data: it carries the long-format
    arrays (y, genus_idx) plus the per-row covariates already broadcast to that layout.
    Each genus gets its own partially pooled slope, so noisy genera shrink toward the
    group while well-measured ones keep their own estimate.

    method picks the inference engine:
      "nuts" gives the publication-quality posterior (use for final, citable runs).
      "advi" is a fast variational approximation for iterating on model structure;
             it underestimates uncertainty in hierarchical models, so don't take its
             HDIs to the manuscript.
    """
    y = model_data["y"]
    genus_idx = model_data["genus_idx"]
    marker = model_data["marker"]
    age = model_data["age"]
    bmi = model_data["bmi"]
    sex = model_data["sex"]
    genus_names = model_data["genus_names"]

    coords = {"genus": genus_names}

    with pm.Model(coords=coords) as model:
        # Each predictor gets the same hierarchical treatment via a closure:
        # a group mean, a group spread, and genus-specific non-centered offsets.
        # Non-centered (mu + tau * z) keeps NUTS healthy when a group spread is small.
        def hierarchical_slope(name):
            group_mean = pm.Normal(f"mu_{name}", 0.0, 1.0)
            group_spread = pm.HalfNormal(f"tau_{name}", 1.0)
            offsets = pm.Normal(f"z_{name}", 0.0, 1.0, dims="genus")
            return pm.Deterministic(
                name, group_mean + group_spread * offsets, dims="genus"
            )

        # Genus baseline abundance on the log scale, partially pooled.
        intercept = hierarchical_slope("intercept")

        # The slope we actually report: adjusted association of the marker per genus.
        beta_marker = hierarchical_slope("beta_marker")

        # Confounders adjusted exactly as in the original GLM, also pooled.
        beta_age = hierarchical_slope("beta_age")
        beta_bmi = hierarchical_slope("beta_bmi")
        beta_sex = hierarchical_slope("beta_sex")

        # Linear predictor on the log scale (NB uses a log link).
        log_mu = (
            intercept[genus_idx]
            + beta_marker[genus_idx] * marker
            + beta_age[genus_idx] * age
            + beta_bmi[genus_idx] * bmi
            + beta_sex[genus_idx] * sex
        )

        # Per-genus overdispersion: rare genera are noisier than abundant ones.
        dispersion = pm.HalfNormal("dispersion", 1.0, dims="genus")

        pm.NegativeBinomial(
            "y_obs",
            mu=pm.math.exp(log_mu),
            alpha=dispersion[genus_idx],
            observed=y,
        )

        if method == "advi":
            # Fast variational preview: minutes, not hours. We pass the seed through
            # so the draw from the fitted approximation is reproducible.
            approx = pm.fit(
                n=advi_iterations,
                method="advi",
                random_seed=random_seed,
            )
            idata = approx.sample(advi_draws)
        elif method == "fullrank_advi":
            approx = pm.fit(
                n=advi_iterations,
                method="fullrank_advi",
                random_seed=random_seed,
            )
            idata = approx.sample(advi_draws)
        elif method == "nuts":
            idata = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                random_seed=random_seed,
                nuts_sampler=nuts_sampler,
                chains=3,
            )
        else:
            raise ValueError(
                f"method must be 'nuts', 'advi', or 'fullrank_advi', got {method!r}"
            )

    return idata
```

**PyMC parametrization note:** `pm.NegativeBinomial(mu=..., alpha=...)` uses the
mean/dispersion parametrization, `Var(y) = mu + mu**2 / alpha`. Larger `alpha` →
closer to Poisson; smaller `alpha` → more overdispersion.

### 3.4 `summarize_marker_associations`: posterior -> ranked table

```python
import arviz as az


def summarize_marker_associations(idata, genus_names, rope_log=0.05, hdi_prob=0.94):
    """Turn the per-genus beta_marker posterior into a ranked, reportable table.

    rope_log defines a "practically null" band on the log scale: |effect| < rope_log
    counts as no meaningful association. 0.05 on the log scale is roughly a 5%
    change in abundance per SD of the marker, so tune it to what's biologically trivial.
    """
    # posterior draws for the genus-specific marker slopes: shape (chains, draws, genus)
    beta = idata.posterior["beta_marker"]

    # Flatten chains and draws so each genus has one long vector of samples.
    samples = (
        beta.stack(sample=("chain", "draw")).transpose("genus", "sample").to_numpy()
    )

    median = np.median(samples, axis=1)
    hdi = az.hdi(idata, var_names=["beta_marker"], prob=hdi_prob)[
        "beta_marker"
    ].to_numpy()

    # Probability of direction: how lopsided the posterior is around zero. Ranges
    # 0.5 (no evidence of a direction) to 1.0 (all mass on one side). This is the
    # Bayesian analog to "is the effect non-zero", replacing the FDR p-value.
    prob_positive = np.mean(samples > 0, axis=1)
    prob_direction = np.maximum(prob_positive, 1 - prob_positive)

    # Fraction of posterior mass outside the practically-null band. High means the
    # effect is not just non-zero but large enough to care about.
    prob_outside_rope = np.mean(np.abs(samples) > rope_log, axis=1)

    table = pd.DataFrame(
        {
            "genus": genus_names,
            # log scale: effect per 1 SD increase in the marker, adjusted for covariates
            "log_effect": median,
            f"hdi_{int(hdi_prob*100)}_low": hdi[:, 0],
            f"hdi_{int(hdi_prob*100)}_high": hdi[:, 1],
            # multiplicative version: exp(log_effect) is the fold-change in abundance
            "fold_change": np.exp(median),
            "prob_direction": prob_direction,
            "prob_outside_rope": prob_outside_rope,
        }
    )

    # A genus is "credible" when the HDI excludes zero: the cleanest one-line call.
    hdi_low = table[f"hdi_{int(hdi_prob*100)}_low"]
    hdi_high = table[f"hdi_{int(hdi_prob*100)}_high"]
    table["hdi_excludes_zero"] = (hdi_low > 0) | (hdi_high < 0)

    # Rank by strength of evidence first, then by effect size, strongest at the top.
    table = table.sort_values(
        ["prob_direction", "log_effect"],
        key=lambda col: col.abs() if col.name == "log_effect" else col,
        ascending=False,
    ).reset_index(drop=True)

    return table
```

### 3.5 End-to-end driver (as used to generate the manuscript table)

```python
import os
import re

selected = [
    "BP.sys", "calpro_final.ug.per.g", "CRP_mgL", "WBC_mm3", "HbA1c.",
    "Hemoglobine_gdL", "GPT_UL", "Bilirubine_dir_mgdL", "Triglyceriden_mgdL",
    "LDL.chol_gem_mgdL", "GFR_CKD-EPI", "Insuline_nuchter_mUL", "health_penalty",
]

tables = []
for s in selected:
    safe_s = re.sub(r"[^\w.\-]", "_", s)
    output_excel = f"./final_data/{safe_s}_genus_associations.xlsx"
    if os.path.exists(output_excel):
        table = pd.read_excel(output_excel)
    else:
        data = build_model_data(merged_df, genera, s)
        idata = fit_genus_marker_model(data, method="nuts")
        table = summarize_marker_associations(idata, data["genus_names"])
        table["feature"] = s
        table.to_excel(output_excel, index=False)
    tables.append(table)

final_table = pd.concat(tables, ignore_index=True)
```

This loop is *usage*, not part of the model itself — included so the developer can
see how the three functions compose, how per-marker results are cached to disk, and
how a `"feature"` column is stamped on for downstream pivoting/plotting (the
plotting/upset-plot code that follows in the notebook is manuscript-specific and is
**not** part of this handoff).

---

## 4. Input data: shape and preparation

### 4.1 The wide participant table (`merged_df`)

One row per participant, columns:

| column | type | notes |
|---|---|---|
| `<marker_name>` | float | the blood/clinical marker being tested, e.g. `CRP_mgL`, `HbA1c.` |
| `age` | float | years |
| `BMI` | float | kg/m² |
| `gender` | **mixed** `"1"` (str) / `-1` (int) | see quirk below — needs cleanup |
| `g_<Genus>`, `uc_f_<Family>`, ... | int | raw genus-level read counts, **rarefied to even depth** |

In the source notebook, `merged_df` is built by:

```python
merged_df["gender"] = merged_df["gender"].apply(lambda x: "1" if x == "m" else -1)
genera = [g for g in merged_df.columns.tolist() if str(g).startswith("g_")]
```

In this dataset: `P ≈ 2341` participants, `G = 236` genus columns → `N ≈ 552,476`
rows in long format (some markers have additional missingness, e.g. `HbA1c.` →
`N = 526,044`).

### 4.2 `model_data` (output of `build_model_data`)

A dict of 1-D numpy arrays, all length `N = P_complete * G`, plus the genus name
list:

| key | shape | meaning |
|---|---|---|
| `y` | `(N,)` float | raw genus counts (observed variable) |
| `genus_idx` | `(N,)` int, `0..G-1` | which genus each row belongs to |
| `marker` | `(N,)` float | z-scored marker value, repeated `G` times per participant |
| `age` | `(N,)` float | z-scored age |
| `bmi` | `(N,)` float | z-scored BMI |
| `sex` | `(N,)` float, `{0,1}` | encoded sex |
| `genus_names` | `list[str]`, len `G` | ordered genus names, used as PyMC coords |

### 4.3 Worked toy example (verified by running the actual code above)

Input (3 participants, 3 genera):

```python
toy = pd.DataFrame(
    {
        "HbA1c.": [5.1, 5.7, 6.2],
        "age":    [60, 57, 48],
        "BMI":    [22.3, 22.9, 23.1],
        "gender": ["1", -1, "1"],   # "1" == male, -1 == female (see §6.2)
        "g_Bacteroides":      [120, 80, 95],
        "g_Prevotella":       [3, 200, 10],
        "g_Faecalibacterium": [60, 45, 70],
    },
    index=["P1", "P2", "P3"],
)
genera = ["g_Bacteroides", "g_Prevotella", "g_Faecalibacterium"]
data = build_model_data(toy, genera, "HbA1c.")
```

Resulting `model_data` (participant-major, genus is the fast axis):

```python
{
 "y":          [120.,   3.,  60.,  80., 200.,  45.,  95.,  10.,  70.],
 "genus_idx":  [  0,    1,    2,    0,    1,    2,    0,    1,    2],
 "marker":     [-1.260, -1.260, -1.260,  0.074,  0.074,  0.074,  1.186,  1.186,  1.186],
 "age":        [ 0.981,  0.981,  0.981,  0.392,  0.392,  0.392, -1.373, -1.373, -1.373],
 "bmi":        [-1.373, -1.373, -1.373,  0.392,  0.392,  0.392,  0.981,  0.981,  0.981],
 "sex":        [ 1.,     1.,     1.,     0.,     0.,     0.,     1.,     1.,     1.],
 "genus_names": ["g_Bacteroides", "g_Prevotella", "g_Faecalibacterium"],
}
```

(With only 3 genera and 3 participants this is far too small to actually fit — it's
here purely to make the wide→long contract unambiguous.)

---

## 5. Example output (real results from the manuscript run)

`summarize_marker_associations` returns one row per genus. Real output for
`calpro_final.ug.per.g` (fecal calprotectin), top rows with `hdi_excludes_zero ==
True`, `G = 236`:

| genus | log_effect | hdi_94_low | hdi_94_high | fold_change | prob_direction | prob_outside_rope | hdi_excludes_zero |
|---|---|---|---|---|---|---|---|
| g_Streptococcus | 0.844 | 0.729 | 0.966 | 2.326 | 1.000 | 1.0000 | True |
| g_Escherichia/Shigella | 0.643 | 0.441 | 0.850 | 1.902 | 1.000 | 1.0000 | True |
| g_Peptoniphilus | 0.618 | 0.456 | 0.778 | 1.855 | 1.000 | 1.0000 | True |
| g_Lactobacillus | 0.593 | 0.378 | 0.846 | 1.809 | 1.000 | 1.0000 | True |
| g_Parvimonas | 0.527 | 0.327 | 0.743 | 1.694 | 1.000 | 1.0000 | True |
| g_Cerasicoccus | -0.328 | -0.492 | -0.165 | 0.720 | 1.000 | 1.0000 | True |
| g_Subdoligranulum | 0.294 | 0.171 | 0.434 | 1.342 | 1.000 | 1.0000 | True |
| g_Collinsella | 0.146 | 0.074 | 0.221 | 1.157 | 1.000 | 0.9985 | True |

And `HbA1c.` (top rows, none reach `hdi_excludes_zero` — illustrates the "mostly
null" regime that the hierarchical shrinkage is designed for):

| genus | log_effect | hdi_94_low | hdi_94_high | fold_change | prob_direction | prob_outside_rope | hdi_excludes_zero |
|---|---|---|---|---|---|---|---|
| g_Bifidobacterium | -0.0112 | -0.0475 | 0.0122 | 0.9889 | 0.867 | 0.038 | False |
| g_Ruminococcus | -0.0104 | -0.0470 | 0.0071 | 0.9896 | 0.865 | 0.017 | False |
| g_Adlercreutzia | -0.0107 | -0.0490 | 0.0171 | 0.9894 | 0.859 | 0.045 | False |

The driver loop writes one such table per marker to
`./final_data/<marker>_genus_associations.xlsx`, then concatenates all of them
(with a `feature` column) into `genus_marker_associations_nuts.xlsx`.

### Posterior variables available on `idata`

For reference, the fitted `idata` exposes (all with `dims=("chain","draw","genus")`
unless noted):

- `mu_intercept`, `tau_intercept`, `z_intercept`, `intercept`
- `mu_beta_marker`, `tau_beta_marker`, `z_beta_marker`, `beta_marker` ← reported
- `mu_beta_age`, `tau_beta_age`, `z_beta_age`, `beta_age`
- `mu_beta_bmi`, `tau_beta_bmi`, `z_beta_bmi`, `beta_bmi`
- `mu_beta_sex`, `tau_beta_sex`, `z_beta_sex`, `beta_sex`
- `dispersion` (dims `("chain","draw","genus")`, not pooled)

`mu_*`/`tau_*` (no `genus` dim) are useful for a quick "is there *any* population-
level signal" check before drilling into per-genus results.

---

## 6. What needs to be generalized (decisions for the package developer)

The user-facing ask was specifically: **age, BMI and sex are currently handled in
an ad-hoc, dataset-specific way and should be redesigned as part of moving this into
the shared package.** The two functions affected are `build_model_data` (encoding)
and `fit_genus_marker_model` (the hardcoded `beta_age`/`beta_bmi`/`beta_sex` terms).
Below is the relevant context to make that design call; the actual design is left to
the developer.

### 6.1 Current shape of the "covariates" concept

- `build_model_data(..., covariates=("age", "BMI", "gender"))` is positional-tuple,
  exactly 3 entries, and *assumes* the first two are continuous (z-scored via
  `standardize`) and the third is sex-like (via `encode_sex`).
- `fit_genus_marker_model` then has **5 hardcoded named terms** (`intercept`,
  `beta_marker`, `beta_age`, `beta_bmi`, `beta_sex`), each built with the same
  `hierarchical_slope(name)` closure and summed into `log_mu`.

A natural generalization: replace the fixed `age`/`bmi`/`sex` triple with an
arbitrary list of covariates, each with a declared encoding (continuous → z-score,
binary → 0/1, categorical with k>2 levels → either reject or one-hot with k-1
extra hierarchical slopes per genus). `hierarchical_slope` already generalizes
trivially — the loop just needs to run over `["marker"] + covariate_names` and
`log_mu` becomes `intercept[idx] + sum(beta[name][idx] * data[name] for name in
[...])`.

### 6.2 The `sex`/`gender` encoding is currently fragile

In the notebook, `merged_df["gender"]` is pre-mapped to a **mixed-type** column —
`"1"` (string) for male, `-1` (int) for female:

```python
merged_df["gender"] = merged_df["gender"].apply(lambda x: "1" if x == "m" else -1)
```

`encode_sex` then does `series.astype("category").cat.codes`. For this specific
mixed `{-1, "1"}` column, pandas happens to order categories as `[-1, "1"]` →
codes `{-1: 0, "1": 1}`, so male ends up as `1` and female as `0` — but **this
relies on incidental pandas category-ordering behavior for mixed-type columns**,
not on anything declared. A different input encoding (e.g. `"M"/"F"` strings, or
`0/1` already) would silently produce a *different* reference level, flipping the
sign of `beta_sex` without any error.

For the package version, this should become an explicit, documented mapping
supplied by the caller (e.g. `sex_reference_level="female"` or a `{value: 0/1}`
dict), not inferred from category order. `encode_sex`'s numeric/boolean
fast-paths are fine to keep.

### 6.3 Missingness drops whole participants

```python
needed = [marker_name, age_col, bmi_col, sex_col]
df = merged_df.dropna(subset=needed).reset_index(drop=True)
```

Any participant missing **any** covariate loses *all G of their genus rows*. With
only 3 covariates this is a modest loss (~2402 → ~2341 participants here). If the
generalized covariate set grows, this could become a much bigger loss — worth
flagging as a tradeoff (vs. e.g. imputing covariates, or fitting separate models
per covariate set).

### 6.4 Other things noticed while extracting this (lower priority)

- `dispersion` is **not** hierarchically pooled (independent `HalfNormal(1)` per
  genus), unlike every other per-genus parameter. Possibly intentional (rare genera
  need their own overdispersion), but worth a deliberate decision rather than an
  accident of the original code.
- No sequencing-depth offset is included in `log_mu`. This is correct **for this
  dataset** because counts come from a table already rarefied to even depth
  (`EvenSampleDepth_genus_matrix.tsv`). A generic package version that accepts
  arbitrary count tables should support an optional `log(total_count)` offset term
  for datasets that are *not* depth-normalized.
- `fit_genus_marker_model`'s `advi_iterations`/`advi_draws` defaults are passed
  through even when `method="nuts"` (harmless, just dead args in that branch — the
  driver loop does this).
- `rope_log=0.05` and `hdi_prob=0.94` in `summarize_marker_associations` are
  reasonable defaults but are domain choices (≈5% abundance change per 1 SD of the
  marker); expose them as tunable parameters (they already are) and document the
  rationale (already done in the docstring).

---

## 7. Performance notes

- `nuts_sampler="numpyro"` + JAX with CUDA is used to run NUTS on GPU
  (`environment.yml` installs `jax[cuda12]` and `numpyro`). On CPU this will be
  much slower — for `G≈236` genera × `N≈550k` rows, NUTS with `draws=500,
  tune=500, chains=3` is a non-trivial job; budget accordingly or default to fewer
  draws/chains for CI/tests.
- `method="advi"` / `"fullrank_advi"` give a fast (~minutes) preview but
  **underestimate posterior uncertainty for hierarchical models** — useful for
  iterating on model structure, not for final HDIs/credible-association calls.
- The driver loop caches one Excel file per marker
  (`./final_data/<marker>_genus_associations.xlsx`) and skips re-fitting if it
  exists — useful pattern to keep for expensive NUTS runs during development.

---

## 8. Suggested README section for the target package

> ## Genus–Marker Hierarchical Negative-Binomial Correlation Model
>
> This model tests, for every taxon in a microbiome count table, whether its
> abundance is associated with a continuous marker (a blood/clinical measurement,
> a phenotype score, etc.), while adjusting for a set of covariates — all in a
> single joint fit.
>
> **Why one joint model instead of one GLM per taxon?**
> Running an independent GLM per taxon and then applying an FDR correction is the
> standard approach, but it treats each taxon's estimate independently and then
> bluntly thresholds the result. This model instead places a **hierarchical
> (partial-pooling) prior** on every per-taxon coefficient: each taxon's
> intercept and slopes are drawn from a shared population distribution
> (`coef[taxon] = mu + tau * z[taxon]`). Taxa with little data or noisy signal are
> automatically pulled toward the population mean ("shrinkage"), while taxa with
> strong, consistent signal keep their own estimate. This is a principled,
> data-driven alternative to ad-hoc multiple-testing correction, and it gives a
> full posterior per taxon instead of a point estimate + p-value.
>
> **Why Negative Binomial?**
> Microbiome read counts are non-negative, integer, and overdispersed relative to
> Poisson (variance >> mean, especially for rare taxa). The NB likelihood with a
> log link and a per-taxon dispersion parameter handles both properties without
> transforming the counts. *If your count table is not already normalized to an
> even sequencing depth per sample, add a `log(total_count)` offset to the linear
> predictor* — the version documented here assumes even-depth input and omits it.
>
> **Why non-centered parametrization?**
> For most marker/taxon combinations, the true association is at or near zero,
> i.e. the population-level spread (`tau`) of that coefficient is small. Sampling
> `coef[taxon] ~ Normal(mu, tau)` directly ("centered") causes funnel-shaped
> posteriors that NUTS struggles with when `tau` is small. The non-centered form
> (`mu + tau * z[taxon]`, `z[taxon] ~ Normal(0,1)`) decouples the geometry and
> keeps sampling efficient and well-mixed.
>
> **Why standardize covariates?**
> Continuous covariates (the marker, age, BMI, ...) are z-scored before fitting.
> This puts all slopes on the same "per 1 SD" scale, makes the `Normal(0,1)` /
> `HalfNormal(1)` priors weakly informative regardless of the covariate's
> original units, and makes slopes directly comparable across taxa and across
> covariates.
>
> **Interpreting the output table:**
> - `log_effect`: posterior median slope, on the log-abundance scale, per 1 SD
>   increase in the marker, adjusted for covariates.
> - `fold_change = exp(log_effect)`: multiplicative change in abundance per 1 SD
>   of the marker.
> - `hdi_*_low` / `hdi_*_high`: highest-density interval of the posterior at the
>   requested probability (default 94%).
> - `prob_direction`: posterior probability mass on the side of zero where most
>   of the posterior lies (0.5 = no evidence of direction, 1.0 = certain
>   direction). The Bayesian analog of "this is non-zero".
> - `prob_outside_rope`: fraction of the posterior outside a "region of practical
>   equivalence" around zero (`rope_log`, default 0.05 on the log scale ≈ a 5%
>   abundance change). High values mean the effect isn't just non-zero, it's
>   large enough to matter.
> - `hdi_excludes_zero`: the simple, conservative "is this credible" flag used for
>   ranking/reporting — `True` when the HDI doesn't straddle zero.
>
> **Fitting:** use `method="nuts"` (via `numpyro`, GPU-accelerated if available)
> for any result that will be reported. `method="advi"`/`"fullrank_advi"` are
> fast variational previews for model-development iteration only — they
> systematically underestimate uncertainty in hierarchical models.
>
> **Covariates (age/BMI/sex and beyond):** see the model docstring for how
> continuous vs. categorical covariates are encoded, and make sure any binary
> covariate's reference level (which value maps to 0 vs. 1) is explicitly
> declared rather than inferred — the sign of its coefficient depends on it.

---

## 9. File/data inventory (for reproducing or testing)

- Genus counts: `data/EvenSampleDepth_genus_matrix.tsv` (rarefied counts, columns
  renamed via `data/EvenSampleDepth_taxonomy.tsv` level-6 names → `g_<Genus>` /
  `uc_f_<Family>` etc.)
- Participant metadata incl. markers/covariates:
  `notebooks/final_data/vdp_metadata_complete.xlsx`
- Per-marker results (already computed, real example outputs):
  `notebooks/final_data/<marker>_genus_associations.xlsx`
- Combined table: `notebooks/final_data/genus_marker_associations_nuts.xlsx`
- Environment: `environment.yml` (`python=3.12`, `pymc`, `nutpie`,
  `jax[cuda12]`, `numpyro`)
