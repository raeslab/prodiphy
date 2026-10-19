# ProDir

The ProDir model can be used to compare the prevalence of specific classes in two populations. E.g. two locations with
a similar ecosystem, one exposed to a pollutant and the other not. Counts, how many individuals per species where 
observed can be provided to the model, and the results will show if the model is confident there is a difference in 
prevalence of that species in the polluted vs unpolluted ecosystem.

## Mathematical formulation

Assume there are $K$ classes and two groups $g \in \{1,2\}$. Define

$$
\mathbf{x}_g = (x_{g1}, \dots, x_{gK})
$$
$$
N_g = \sum_{k=1}^{K} x_{gk}
$$
$$
\mathbf{p}_g = (p_{g1}, \dots, p_{gK})
$$
$$
\sum_{k=1}^{K} p_{gk} = 1
$$

Here, $\mathbf{x}_g$ is the observed count vector in group $g$, $N_g$ is the total sample size,
and $\mathbf{p}_g$ is the latent class-prevalence vector.

Using a symmetric Dirichlet prior and a multinomial observation model:

$$
\mathbf{p}_g \sim \mathrm{Dirichlet}(\mathbf{1})
$$
$$
\mathbf{x}_g \mid \mathbf{p}_g \sim \mathrm{Multinomial}(N_g, \mathbf{p}_g)
$$

By Dirichlet--multinomial conjugacy, the posterior is:

$$
\mathbf{p}_g \mid \mathbf{x}_g \sim
\mathrm{Dirichlet}(\mathbf{x}_g + \mathbf{1})
$$

This is exactly what the implementation samples from via:

$$
\mathbf{p}_1 \sim \mathrm{Dirichlet}(\mathbf{x}_1 + \mathbf{1})
$$
$$
\mathbf{p}_2 \sim \mathrm{Dirichlet}(\mathbf{x}_2 + \mathbf{1})
$$

For each class $k$, ProDir reports posterior draws of three effect-size quantities:

$$
\Delta_k = p_{1k} - p_{2k}
$$
$$
R_k = \log_2\left(\frac{p_{2k}}{p_{1k}}\right)
$$
$$
\mathrm{LOR}_k = \log\left(\frac{p_{2k}}{1 - p_{2k}}\right) - \log\left(\frac{p_{1k}}{1 - p_{1k}}\right)
$$

$`\mathrm{LOR}_k`$ is the natural log of the one-vs-rest odds ratio (odds of class $k$ versus all other classes in
group 2 compared to group 1). It is reported as `log_odds_ratio_<label>` and back-transformed as `odds_ratio_<label>`.
The $+1$ pseudo-count keeps every $`p_{gk}`$ strictly between 0 and 1, so the odds ratio is finite in every draw, also for
classes with zero counts in one of the groups (similar in spirit to the Haldane-Anscombe $+0.5$ correction).

Inference is performed by sampling from the posterior using PyMC's sampler; summary statistics (mean, SD, and HDI intervals)
for $\Delta_k$, $R_k$, $`\mathrm{LOR}_k`$, the odds ratio and each $`p_{gk}`$ are returned.

## Example Usage

```python
from prodiphy import ProDir

if __name__ == "__main__":
    reference_counts = [100, 30, 20, 10]
    polluted_counts = [90, 20, 5, 10]
    
    labels = ["SpeciesA", "SpeciesB", "SpeciesC", "SpeciesD"]
    
    model = ProDir()
    model.fit(reference_counts, polluted_counts, labels)
    
    summary = model.get_stats()
```

## Example Output


When looking at results in `summary` below, we see the estimated proportion for each species in both ecosystems in 
`group_1_p_SpeciesA`, `group_2_p_SpeciesA`, `group_1_p_SpeciesB`, ... along with the uncertainty on the observation. We
also get differences between the two groups as a delta, log2 ratio and one-vs-rest odds ratio, again with the uncertainty
on those values. E.g. here the prevalence of Species C is confidently decreased as the HDI on `log2_ratio_SpeciesC` is
between -2.843 and -0.3 (note that zero, which indicates no difference, is not in the interval). The odds ratio tells the
same story: `odds_ratio_SpeciesC` is 0.353 with an HDI between 0.077 and 0.678, so one is not in the interval.

|                         |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:------------------------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| delta_SpeciesA          | -0.091 | 0.055 |   -0.196 |     0.006 |       0.001 |     0.001 |       5298 |       3477 |       1 |
| group_1_p_SpeciesA      |  0.615 | 0.038 |    0.545 |     0.688 |       0.001 |     0.001 |       4806 |       2381 |       1 |
| group_2_p_SpeciesA      |  0.706 | 0.04  |    0.629 |     0.779 |       0.001 |     0.001 |       5295 |       3302 |       1 |
| log2_ratio_SpeciesA     |  0.2   | 0.121 |   -0.023 |     0.424 |       0.002 |     0.002 |       5158 |       3423 |       1 |
| log_odds_ratio_SpeciesA |  0.413 | 0.25  |   -0.03  |     0.902 |       0.003 |     0.004 |       5390 |       3477 |       1 |
| odds_ratio_SpeciesA     |  1.559 | 0.395 |    0.861 |     2.274 |       0.005 |     0.007 |       5390 |       3477 |       1 |
| delta_SpeciesB          |  0.027 | 0.045 |   -0.058 |     0.112 |       0.001 |     0.001 |       4639 |       2935 |       1 |
| group_1_p_SpeciesB      |  0.189 | 0.031 |    0.134 |     0.248 |       0     |     0.001 |       4246 |       2784 |       1 |
| group_2_p_SpeciesB      |  0.162 | 0.033 |    0.101 |     0.222 |       0     |     0.001 |       4525 |       3065 |       1 |
| log2_ratio_SpeciesB     | -0.235 | 0.379 |   -0.937 |     0.502 |       0.006 |     0.006 |       4545 |       3076 |       1 |
| log_odds_ratio_SpeciesB | -0.196 | 0.317 |   -0.773 |     0.426 |       0.005 |     0.005 |       4582 |       3054 |       1 |
| odds_ratio_SpeciesB     |  0.865 | 0.28  |    0.369 |     1.361 |       0.004 |     0.005 |       4582 |       3054 |       1 |
| delta_SpeciesC          |  0.082 | 0.033 |    0.018 |     0.14  |       0     |     0     |       4326 |       3276 |       1 |
| group_1_p_SpeciesC      |  0.128 | 0.026 |    0.078 |     0.176 |       0     |     0     |       4429 |       3012 |       1 |
| group_2_p_SpeciesC      |  0.046 | 0.018 |    0.016 |     0.081 |       0     |     0     |       3098 |       2715 |       1 |
| log2_ratio_SpeciesC     | -1.546 | 0.675 |   -2.843 |    -0.3   |       0.012 |     0.011 |       3499 |       2818 |       1 |
| log_odds_ratio_SpeciesC | -1.162 | 0.499 |   -2.138 |    -0.251 |       0.008 |     0.008 |       3552 |       2839 |       1 |
| odds_ratio_SpeciesC     |  0.353 | 0.178 |    0.077 |     0.678 |       0.003 |     0.003 |       3552 |       2839 |       1 |
| delta_SpeciesD          | -0.018 | 0.031 |   -0.082 |     0.037 |       0     |     0     |       5456 |       2946 |       1 |
| group_1_p_SpeciesD      |  0.068 | 0.019 |    0.034 |     0.105 |       0     |     0     |       5458 |       3259 |       1 |
| group_2_p_SpeciesD      |  0.085 | 0.024 |    0.043 |     0.131 |       0     |     0     |       5043 |       3106 |       1 |
| log2_ratio_SpeciesD     |  0.339 | 0.604 |   -0.807 |     1.484 |       0.008 |     0.008 |       5319 |       3675 |       1 |
| log_odds_ratio_SpeciesD |  0.254 | 0.452 |   -0.594 |     1.12  |       0.006 |     0.006 |       5348 |       3747 |       1 |
| odds_ratio_SpeciesD     |  1.428 | 0.687 |    0.393 |     2.639 |       0.01  |     0.016 |       5348 |       3747 |       1 |
