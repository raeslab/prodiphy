# DeltaSlopeHierarchical

A variant of [DeltaSlope](deltaslope.md) that compares the slope, intercept and spread of a linear regression between
two groups, but with **hierarchical shrinkage**: the slopes and intercepts of both groups are drawn from a shared
population distribution, so differences between groups are pulled towards zero unless the data supports them. This
makes the estimated deltas more conservative, which is useful when groups are small or noisy.

The API is identical to DeltaSlope: create the model, call `fit()` with a reference and target DataFrame and the
names of the x and y columns, and use `get_stats()` to get the results.

## Mathematical formulation

Let observations be indexed by $i = 1, \dots, n$, with predictor $x_i$, response $y_i$, and group indicator
$g_i \in \{0,1\}$ where $g_i=0$ denotes the reference group and $g_i=1$ the target group.

The model is fitted on standardized data, using the mean and standard deviation of the pooled data of both groups:

$$
\tilde{x}_i = \frac{x_i - \bar{x}}{s_x}, \qquad \tilde{y}_i = \frac{y_i - \bar{y}}{s_y}
$$

The likelihood is

$$
\tilde{y}_i \sim \mathrm{Normal}(\beta_{g_i} \tilde{x}_i + \alpha_{g_i}, \sigma_{g_i})
$$

with non-centered hierarchical priors on the slopes and intercepts ($g \in \{0, 1\}$)

$$
\beta_g = \mu_\beta + z_{\beta,g} \tau_\beta, \qquad z_{\beta,g} \sim \mathrm{Normal}(0, 1)
$$
$$
\alpha_g = \mu_\alpha + z_{\alpha,g} \tau_\alpha, \qquad z_{\alpha,g} \sim \mathrm{Normal}(0, 1)
$$
$$
\mu_\beta, \mu_\alpha \sim \mathrm{Normal}(0, 1)
$$
$$
\tau_\beta \sim \mathrm{HalfNormal}(s_\beta), \qquad \tau_\alpha \sim \mathrm{HalfNormal}(s_\alpha)
$$
$$
\sigma_g \sim \mathrm{HalfNormal}(1)
$$

where $s_\beta$ and $s_\alpha$ are set with the `sigma_slope_prior` and `sigma_intercept_prior` arguments
(both default to 0.5). Because $\tilde{x}$ is centered, $\alpha_g$ is the expected (standardized) value of $y$ at the
pooled mean of $x$.

All reported parameters are transformed back to the original scale:

$$
\mathrm{slope}_g = \beta_g \frac{s_y}{s_x}, \qquad
\mathrm{at\_mean\_x}_g = \bar{y} + \alpha_g s_y, \qquad
\mathrm{intercept}_g = \mathrm{at\_mean\_x}_g - \mathrm{slope}_g \bar{x}, \qquad
\mathrm{sigma}_g = \sigma_g s_y
$$

and every delta is defined as target minus reference, e.g.
$\Delta_{\mathrm{slope}} = \mathrm{slope}_1 - \mathrm{slope}_0$.

### Notes

* **The strength of shrinkage is mostly set by the prior.** With only two groups, the data contain very little
  information about the between-group standard deviations $\tau_\beta$ and $\tau_\alpha$. Their priors
  (`sigma_slope_prior`, `sigma_intercept_prior`, on the standardized scale) therefore effectively determine how
  strongly the groups are pulled together. Lower values give stronger shrinkage.
* **Intercepts are shrunk at the mean of x, not at x = 0.** The intercept at $x = 0$ is often far outside the
  observed data, so `delta_at_mean_x` (the difference between groups in the fitted value of $y$ at the pooled mean of
  $x$) is usually the more interpretable quantity.
* **Sampling.** `target_accept` defaults to 0.95 (it can be overridden through `sample_kwargs`). The non-centered
  parameterization works best for small or noisy groups. With large, clean datasets a small number of divergences
  can occur; check the `r_hat` and `ess` columns in the output.

## Example Usage

```python
from prodiphy import DeltaSlopeHierarchical
import pandas as pd
import numpy as np


def build_dataset():
    x_ref = np.random.randint(0, 100, size=200)
    y_ref = x_ref * 2 + 10 + np.random.normal(0, 2, size=200)

    x_target = np.random.randint(0, 100, size=200)
    y_target = x_target * 1.5 + 9 + np.random.normal(0, 3, size=200)

    ref_df = pd.DataFrame({"x": x_ref, "y": y_ref})
    target_df = pd.DataFrame({"x": x_target, "y": y_target})

    return ref_df, target_df


if __name__ == "__main__":
    ref_df, target_df = build_dataset()

    model = DeltaSlopeHierarchical()
    model.fit(ref_df, target_df, x="x", y="y")

    summary = model.get_stats()

    print(summary.to_markdown())
```

## Example Output

Using the `get_stats()` function, after fitting the model, estimates for the intercept, slope, sigma and fitted value
at the mean of x for both groups can be extracted along with the difference (delta, target - reference) between groups.

|                  |    mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:-----------------|--------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| delta_at_mean_x  | -25.493 | 0.269 |  -25.992 |   -24.979 |       0.003 |     0.003 |       8214 |       5150 |       1 |
| delta_intercept  |  -1.289 | 0.523 |   -2.316 |    -0.37  |       0.006 |     0.006 |       8040 |       5930 |       1 |
| delta_sigma      |   1.171 | 0.189 |    0.84  |     1.548 |       0.002 |     0.002 |       7087 |       5474 |       1 |
| delta_slope      |  -0.496 | 0.009 |   -0.513 |    -0.479 |       0     |     0     |       7935 |       5290 |       1 |
| ref_at_mean_x    | 107.455 | 0.148 |  107.168 |   107.729 |       0.002 |     0.002 |       8344 |       5492 |       1 |
| ref_intercept    |  10.068 | 0.288 |    9.547 |    10.625 |       0.003 |     0.003 |       7945 |       6064 |       1 |
| ref_sigma        |   2.026 | 0.102 |    1.839 |     2.216 |       0.001 |     0.001 |       7126 |       5262 |       1 |
| ref_slope        |   1.998 | 0.005 |    1.988 |     2.007 |       0     |     0     |       7694 |       6303 |       1 |
| target_at_mean_x |  81.962 | 0.226 |   81.54  |    82.378 |       0.002 |     0.003 |       8373 |       5589 |       1 |
| target_intercept |   8.779 | 0.44  |    7.979 |     9.639 |       0.005 |     0.005 |       8186 |       6397 |       1 |
| target_sigma     |   3.197 | 0.16  |    2.903 |     3.497 |       0.002 |     0.002 |       7426 |       5750 |       1 |
| target_slope     |   1.501 | 0.008 |    1.487 |     1.516 |       0     |     0     |       7851 |       5383 |       1 |
