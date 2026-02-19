# DeltaSlope

When performing a linear regression between two groups, this model can be used to check if there are differences in
the slope of the regression, the intercept and the spread. The example code below shows how to create two dataframes,
one for each group, and how to run the model specifying columns for the features corresponding to the x and y axes.


## Mathematical formulation

Let observations be indexed by $i = 1, \dots, n$, with predictor $x_i$, response $y_i$, and group indicator
$g_i \in \{0,1\}$ where $g_i=0$ denotes the reference group and $g_i=1$ the target group.

The model used in the implementation is

$$
y_i \sim \mathrm{Normal}(\mu_i, \sigma_{g_i})
$$
$$
\mu_i = \left(\beta_{\mathrm{ref}} + \Delta_\beta g_i\right)x_i + \alpha_{g_i}
$$

with priors

$$
\beta_{\mathrm{ref}} \sim \mathrm{Normal}(0, 2)
$$
$$
\Delta_\beta \sim \mathrm{Normal}(0, 1)
$$
$$
\alpha_0, \alpha_1 \sim \mathrm{Normal}(0, 2)
$$
$$
\sigma_0, \sigma_1 \sim \mathrm{Normal}(1, 1)
$$

From these parameters, DeltaSlope defines derived quantities reported in the summary:

$$
\beta_{\mathrm{target}} = \beta_{\mathrm{ref}} + \Delta_\beta
$$
$$
\Delta_\alpha = \alpha_1 - \alpha_0
$$
$$
\Delta_\sigma = \sigma_1 - \sigma_0
$$

and

$$
\alpha_{\mathrm{ref}} = \alpha_0
$$
$$
\alpha_{\mathrm{target}} = \alpha_1
$$
$$
\sigma_{\mathrm{ref}} = \sigma_0
$$
$$
\sigma_{\mathrm{target}} = \sigma_1
$$

Inference is performed by sampling from the posterior using PyMC's sampler and reporting posterior summaries
(mean, SD, and HDI intervals) for these reference, target, and delta parameters.

## Example Usage

```python
from prodiphy import DeltaSlope
import pandas as pd
import numpy as np


def build_dataset():
    x_ref = np.random.randint(0, 100, size=200)
    y_ref = x_ref * 2 + 10 + np.random.normal(0,2,size=200)

    x_target = np.random.randint(0, 100, size=200)
    y_target = x_target * 1.5 + 9 + np.random.normal(0,3,size=200)

    ref_df = pd.DataFrame({"x": x_ref, "y": y_ref})
    target_df = pd.DataFrame({"x": x_target, "y": y_target})

    return ref_df, target_df


if __name__ == "__main__":
    ref_df, target_df = build_dataset()

    deltaslope = DeltaSlope()
    deltaslope.fit(ref_df, target_df, x="x", y="y")

    summary = deltaslope.get_stats()

    print(summary.to_markdown())
```

## Example Output

Using the `get_stats()` function, after fitting the model, estimates for the intercept, slope and sigma for the
regression for both groups can be extracted along with the difference (delta) between groups.

|                  |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:-----------------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| delta_intercept  | -1.705 | 0.533 |   -2.681 |    -0.684 |       0.008 |     0.006 |       4669 |       4760 |       1 |
| delta_sigma      |  1.164 | 0.181 |    0.819 |     1.501 |       0.002 |     0.002 |       6978 |       5396 |       1 |
| delta_slope      | -0.486 | 0.009 |   -0.503 |    -0.468 |       0     |     0     |       4522 |       4732 |       1 |
| ref_intercept    | 10.107 | 0.296 |    9.562 |    10.667 |       0.004 |     0.003 |       4647 |       4899 |       1 |
| ref_sigma        |  1.973 | 0.1   |    1.795 |     2.167 |       0.001 |     0.001 |       7302 |       5167 |       1 |
| ref_slope        |  2.001 | 0.005 |    1.991 |     2.01  |       0     |     0     |       4361 |       4824 |       1 |
| target_intercept |  8.402 | 0.441 |    7.602 |     9.262 |       0.006 |     0.004 |       5640 |       5263 |       1 |
| target_sigma     |  3.137 | 0.15  |    2.863 |     3.424 |       0.002 |     0.001 |       7386 |       5397 |       1 |
| target_slope     |  1.514 | 0.008 |    1.5   |     1.529 |       0     |     0     |       5635 |       5535 |       1 |

