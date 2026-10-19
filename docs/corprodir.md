# CorProDir

The CorProDir (Corrected Proportional Dirichlet) model extends the basic ProDir model by incorporating additional 
covariates when comparing proportions between two populations. This makes it possible to account for confounding 
factors that might influence the observed differences.

## Mathematical formulation

CorProDir compares category prevalences for a target cohort against prevalences expected from a reference cohort after
adjusting for confounders. Let there be $K$ labels, index $k \in \{1,\dots,K\}$, and confounder vector
$\mathbf{z}_i \in \mathbb{R}^P$ for individual $i$.

Define target counts:

$$
\mathbf{x}^{(T)} = \left(x^{(T)}_1, \dots, x^{(T)}_K\right)
$$
$$
N_T = \sum_{k=1}^{K} x^{(T)}_k
$$

The implementation uses a two-part Bayesian procedure.

1) Uncorrected target prevalence (Dirichlet model)

$$
\mathbf{p}^{(T)} \sim \mathrm{Dirichlet}\left(\mathbf{x}^{(T)} + \mathbf{1}\right)
$$

This yields posterior draws $p^{(T)}_{k,s}$ for each label $k$ and posterior sample index $s$.

2) Confounder-corrected expected prevalence (multinomial regression)

The reference data are modeled with a multinomial GLM (via Bambi):

$$
\Pr\left(y_i = k \mid \mathbf{z}_i\right) = \pi_k\left(\mathbf{z}_i\right)
$$
$$
\mathbf{\pi}(\mathbf{z}_i) = \mathrm{softmax}\left(\eta_1(\mathbf{z}_i), \dots, \eta_K(\mathbf{z}_i)\right)
$$

Where $y_i \in \{1, \dots, K\}$ denotes the observed class label for the $i$-th observation, and
$\mathbf{z}_i$ is the vector of associated confounders. In the Bambi formula $c(\mathrm{labels}) \sim$
confounders, the `labels` column in the data frame corresponds to the collection of observed $y_i$ values.

With linear predictors $\eta_k(\mathbf{z}_i)$ parameterized by the confounders.

For each posterior draw $s$, predicted response probabilities are computed for all target individuals and averaged:

$$
\hat{p}^{(R\to T)}_{k,s} = \frac{1}{n_T} \sum_{i=1}^{n_T} \pi_{k,s}\left(\mathbf{z}^{(T)}_i\right)
$$

where $\hat{p}^{(R\to T)}_{k,s}$ is the prevalence expected in the target cohort under the reference-derived model.

CorProDir then reports draw-wise contrasts:

$$
\Delta_{k,s} = p^{(T)}_{k,s} - \hat{p}^{(R\to T)}_{k,s}
$$
$$
R_{k,s} = \log_2\left(\frac{p^{(T)}_{k,s}}{\hat{p}^{(R\to T)}_{k,s}}\right)
$$
$$
\mathrm{LOR}_{k,s} = \log\left(\frac{p^{(T)}_{k,s}}{1 - p^{(T)}_{k,s}}\right) - \log\left(\frac{\hat{p}^{(R\to T)}_{k,s}}{1 - \hat{p}^{(R\to T)}_{k,s}}\right)
$$

and summarizes them per label with posterior means, standard deviations, HDIs, and posterior sign probabilities
$f_k^+ = \Pr(\Delta_k > 0)$ and $f_k^- = \Pr(\Delta_k < 0)$ estimated from sampled draws.

$`\mathrm{LOR}_{k}`$ is the natural log of the one-vs-rest odds ratio: the odds of label $k$ (versus all other labels)
in the target compared to the odds expected from the reference. It is reported on the log scale (mean, SD, HDI) and
back-transformed as `odds_ratio` $`= \exp(\mathrm{median}(\mathrm{LOR}_k))`$ with the exponentiated HDI bounds.

Because $`p^{(T)}_{k,s}`$ is drawn from a Dirichlet with a $+1$ pseudo-count and $`\hat{p}^{(R\to T)}_{k,s}`$ is an
average of softmax probabilities, both are strictly between 0 and 1, so every draw of $`R_k`$ and $`\mathrm{LOR}_k`$ is
finite, also for labels that are absent from the target.

### Prediction kind

`fit()` accepts `kind="response_params"` (default) or `kind="response"`. The default averages the predicted class
probabilities $`\pi_{k,s}(\mathbf{z}^{(T)}_i)`$ as in the formula above. `kind="response"` instead averages one
simulated outcome per target individual from the posterior predictive distribution. That adds binomial noise to the
expected prevalence and, for rare labels or small targets, gives draws where the expected prevalence is exactly 0,
making $`R_k`$ and $`\mathrm{LOR}_k`$ infinite and the summary statistics unusable. See `comparison.html` in the
repository root for a simulation study comparing both options.

## Key Features

- Accounts for covariates (e.g., age, BMI) when comparing proportions between populations
- Provides estimates of group prevalence differences while controlling for confounding factors
- Returns credible intervals (HDI) for the differences, log2 ratios and one-vs-rest odds ratios
- Calculates the fraction of posterior samples above/below zero to assess the reliability of differences

## Example Usage

```python
import pandas as pd
import numpy as np

from prodiphy import CorProDir

np.random.seed(1910)

labels = ["a", "b", "c", "d"]


def build_data():

    ref_size = 800
    target_size = 100
    ref_prevalence = [0.3, 0.3, 0.2, 0.2]
    target_prevalence = [0.3, 0.2, 0.3, 0.2]

    ref_df = pd.DataFrame(
        {
            "age": np.random.randint(18, high=80, size=ref_size),
            "BMI": np.random.normal(25, size=ref_size),
            "label": np.random.choice(
                labels, size=ref_size, replace=True, p=ref_prevalence
            ),
        }
    )
    target_df = pd.DataFrame(
        {
            "age": np.random.randint(18, high=80, size=target_size),
            "BMI": np.random.normal(25, size=target_size),
            "label": np.random.choice(
                labels, size=target_size, replace=True, p=target_prevalence
            ),
        }
    )

    for label in labels:
        ref_df[label] = ref_df["label"].apply(lambda x: 1 if x == label else 0)
        target_df[label] = target_df["label"].apply(lambda x: 1 if x == label else 0)

    return ref_df, target_df


if __name__ == "__main__":
    ref_df, target_df = build_data()

    model = CorProDir(draws=500)
    model.fit(ref_df, target_df, "label", ["age", "BMI"])

    output = model.get_stats()
    output.to_excel("./tmp/example.xlsx")
```

## Example Output

The output is shown below (note the difference from [ProDir](prodir.md)).

|    | label   |   mean_fraction |   mean_estimate |   mean_delta |   std_delta |   hdi_low_delta |   hdi_high_delta |   mean_log2_ratio |   std_log2_ratio |   hdi_low_log2_ratio |   hdi_high_log2_ratio |   mean_log_odds_ratio |   std_log_odds_ratio |   hdi_low_log_odds_ratio |   hdi_high_log_odds_ratio |   odds_ratio |   hdi_low_odds_ratio |   hdi_high_odds_ratio |   fraction_above_zero |   fraction_below_zero |
|---:|:--------|----------------:|----------------:|-------------:|------------:|----------------:|-----------------:|------------------:|-----------------:|---------------------:|----------------------:|----------------------:|---------------------:|-------------------------:|--------------------------:|-------------:|---------------------:|----------------------:|----------------------:|----------------------:|
|  0 | a       |        0.296428 |        0.304101 |  -0.00767321 |   0.0473689 |     -0.0978177  |        0.0751511 |        -0.0504623 |         0.230479 |           -0.522258  |              0.332867 |            -0.0442523 |             0.227652 |               -0.505333  |                  0.336457 |     0.953854 |             0.603305 |              1.39998  |                 0.396 |                 0.604 |
|  1 | b       |        0.211911 |        0.315808 |  -0.103897   |   0.0449282 |     -0.18955    |       -0.0257471 |        -0.601704  |         0.297468 |           -1.20401   |             -0.101931 |            -0.557346  |             0.263857 |               -1.10008   |                 -0.127399 |     0.573308 |             0.332843 |              0.880383 |                 0.02  |                 0.98  |
|  2 | c       |        0.311321 |        0.212587 |   0.0987334  |   0.0493033 |      0.00406392 |        0.190645  |         0.537932  |         0.24719  |            0.0559062 |              0.998085 |             0.508885  |             0.241306 |                0.0251194 |                  0.945494 |     1.67925  |             1.02544  |              2.57408  |                 0.978 |                 0.022 |
|  3 | d       |        0.18034  |        0.167503 |   0.0128369  |   0.0401871 |     -0.0580957  |        0.0908694 |         0.0780199 |         0.331518 |           -0.541432  |              0.662363 |             0.0705853 |             0.278516 |               -0.441546  |                  0.579344 |     1.08221  |             0.643041 |              1.78487  |                 0.612 |                 0.388 |
