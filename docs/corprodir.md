# CorProDir

The CorProDir (Corrected Proportional Dirichlet) model extends the basic ProDir model by incorporating additional 
covariates when comparing proportions between two populations. This makes it possible to account for confounding 
factors that might influence the observed differences.

## Key Features

- Accounts for covariates (e.g., age, BMI) when comparing proportions between populations
- Provides estimates of group prevalence differences while controlling for confounding factors
- Returns confidence intervals (HDI) for the differences and log2 ratios
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
