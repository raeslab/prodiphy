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

and summarizes them per label with posterior means, standard deviations, HDIs, and posterior sign probabilities
$f_k^+ = \Pr(\Delta_k > 0)$ and $f_k^- = \Pr(\Delta_k < 0)$ estimated from sampled draws.

## Key Features

- Accounts for covariates (e.g., age, BMI) when comparing proportions between populations
- Provides estimates of group prevalence differences while controlling for confounding factors
- Returns credible intervals (HDI) for the differences and log2 ratios
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

|    | label   |   mean_fraction |   mean_estimate |   mean_delta |   std_delta |   hdi_low_delta |   hdi_high_delta |   mean_log2_ratio |   std_log2_ratio |   hdi_low_log2_ratio |   hdi_high_log2_ratio |   fraction_above_zero |   fraction_below_zero |
|---:|:--------|----------------:|----------------:|-------------:|------------:|----------------:|-----------------:|------------------:|-----------------:|---------------------:|----------------------:|----------------------:|----------------------:|
|  0 | d       |        0.184428 |         0.1666  |    0.0178277 |   0.0554936 |     -0.0777508  |        0.125228  |         0.15878   |         0.482594 |            -0.803177 |             1.02433   |                 0.624 |                 0.376 |
|  1 | c       |        0.310888 |         0.20974 |    0.101148  |   0.0584871 |     -0.00760088 |        0.212862  |         0.583754  |         0.351984 |            -0.14581  |             1.18245   |                 0.942 |                 0.058 |
|  2 | a       |        0.293853 |         0.30812 |   -0.0142665 |   0.0671975 |     -0.120816   |        0.118628  |        -0.0670092 |         0.32641  |            -0.624287 |             0.553964  |                 0.402 |                 0.598 |
|  3 | b       |        0.210831 |         0.31554 |   -0.104709  |   0.0607522 |     -0.209621   |        0.0146693 |        -0.586071  |         0.347963 |            -1.20928  |             0.0975963 |                 0.04  |                 0.96  |
