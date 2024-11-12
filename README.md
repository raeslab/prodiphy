[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# ProDiphy

Python package that implements probabilistic models to compare (sub-)populations.

## Installation

Note: currently the package has not been deposited in a repository, so it needs to be installed through github.

To install ProDiphy, [conda]([https://conda.io/projects/conda/en/latest/index.html](https://github.com/conda-forge/miniforge)) is required (or miniconda).

First, clone the repository to create a local copy in the current directory.

```bash
git clone https://github.com/raeslab/prodiphy
```

Then navigate to the source code, create an environment using the yaml file in the folder `docs/dev`, activate it and install the local copy of the code.

```bash
cd prodiphy
conda env create -f docs/dev/environment.yml
conda activate prodiphy
pip install git+https://github.com/bambinos/bambi.git@refs/pull/847/head
pip install -e .
```

Note: currently only works with a development version of bambi, which needs to be installed from a PR on GitHub.

## Usage

### ProDir Model

The ProDir model can be used to compare the prevalence of specific classes in two populations. E.g. two locations with
a similar ecosystem, one exposed to a pollutant and the other not. Counts, how many individuals per species where 
observed can be provided to the model, and the results will show if the model is confident there is a difference in 
prevalence of that species in the polluted vs unpolluted ecosystem.

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

When looking at results in `summary` below, we see the estimated proportion for each species in both ecosystems in 
`group_1_p_SpeciesA`, `group_2_p_SpeciesA`, `group_1_p_SpeciesB`, ... along with the uncertainty on the observation. We
also get differences between the two groups as a delta and log2 ratio, again with the uncertainty on those values.
E.g. here the prevalence of Species C is confidently decreased as the HDI on `log2_ratio_SpeciesC` is between 
-2.833 and -0.342 (note that zero, which indicates no difference, is not in the interval).

|                     |   mean |    sd |   hdi_3% |   hdi_97% |   mcse_mean |   mcse_sd |   ess_bulk |   ess_tail |   r_hat |
|:--------------------|-------:|------:|---------:|----------:|------------:|----------:|-----------:|-----------:|--------:|
| delta_SpeciesA      | -0.091 | 0.055 |   -0.184 |     0.025 |       0.001 |     0.001 |       3914 |       3326 |       1 |
| group_1_p_SpeciesA  |  0.616 | 0.039 |    0.544 |     0.687 |       0.001 |     0     |       4354 |       2532 |       1 |
| group_2_p_SpeciesA  |  0.707 | 0.04  |    0.629 |     0.778 |       0.001 |     0     |       4248 |       3383 |       1 |
| log2_ratio_SpeciesA |  0.2   | 0.123 |   -0.042 |     0.42  |       0.002 |     0.001 |       3904 |       3235 |       1 |
| delta_SpeciesB      |  0.027 | 0.044 |   -0.057 |     0.109 |       0.001 |     0.001 |       4346 |       2858 |       1 |
| group_1_p_SpeciesB  |  0.189 | 0.031 |    0.133 |     0.247 |       0     |     0     |       4300 |       3325 |       1 |
| group_2_p_SpeciesB  |  0.162 | 0.032 |    0.102 |     0.223 |       0     |     0     |       4647 |       3334 |       1 |
| log2_ratio_SpeciesB | -0.231 | 0.375 |   -0.927 |     0.484 |       0.006 |     0.005 |       4277 |       2779 |       1 |
| delta_SpeciesC      |  0.082 | 0.032 |    0.022 |     0.142 |       0.001 |     0     |       3441 |       2950 |       1 |
| group_1_p_SpeciesC  |  0.128 | 0.027 |    0.079 |     0.177 |       0     |     0     |       3803 |       2798 |       1 |
| group_2_p_SpeciesC  |  0.045 | 0.018 |    0.015 |     0.079 |       0     |     0     |       3035 |       2427 |       1 |
| log2_ratio_SpeciesC | -1.577 | 0.676 |   -2.833 |    -0.342 |       0.013 |     0.009 |       2909 |       2657 |       1 |
| delta_SpeciesD      | -0.018 | 0.032 |   -0.079 |     0.041 |       0     |     0     |       4807 |       3317 |       1 |
| group_1_p_SpeciesD  |  0.067 | 0.019 |    0.031 |     0.102 |       0     |     0     |       4462 |       2855 |       1 |
| group_2_p_SpeciesD  |  0.085 | 0.025 |    0.041 |     0.132 |       0     |     0     |       5259 |       2906 |       1 |
| log2_ratio_SpeciesD |  0.344 | 0.618 |   -0.79  |     1.523 |       0.009 |     0.008 |       4506 |       3332 |       1 |

## CorProDir

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

The output is shown below (note the difference from ProDir).

|    | label   |   mean_fraction |   mean_estimate |   mean_delta |   std_delta |   hdi_low_delta |   hdi_high_delta |   mean_log2_ratio |   std_log2_ratio |   hdi_low_log2_ratio |   hdi_high_log2_ratio |   fraction_above_zero |   fraction_below_zero |
|---:|:--------|----------------:|----------------:|-------------:|------------:|----------------:|-----------------:|------------------:|-----------------:|---------------------:|----------------------:|----------------------:|----------------------:|
|  0 | d       |        0.184428 |         0.1666  |    0.0178277 |   0.0554936 |     -0.0777508  |        0.125228  |         0.15878   |         0.482594 |            -0.803177 |             1.02433   |                 0.624 |                 0.376 |
|  1 | c       |        0.310888 |         0.20974 |    0.101148  |   0.0584871 |     -0.00760088 |        0.212862  |         0.583754  |         0.351984 |            -0.14581  |             1.18245   |                 0.942 |                 0.058 |
|  2 | a       |        0.293853 |         0.30812 |   -0.0142665 |   0.0671975 |     -0.120816   |        0.118628  |        -0.0670092 |         0.32641  |            -0.624287 |             0.553964  |                 0.402 |                 0.598 |
|  3 | b       |        0.210831 |         0.31554 |   -0.104709  |   0.0607522 |     -0.209621   |        0.0146693 |        -0.586071  |         0.347963 |            -1.20928  |             0.0975963 |                 0.04  |                 0.96  |

## DeltaSlope

When performing a linear regression between two groups, this model can be used to check if there are differences in
the slope of the regression, the intercept and the spread. The example code below shows how to create two dataframes,
one for each group, and how to run the model specifying columns for the features corresponding to the x and y axes.

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



## Contributing

Any contributions you make are **greatly appreciated**.

  * Found a bug or have some suggestions? Open an issue.
  * Pull requests are welcome! Though open an issue first to discuss which features/changes you wish to implement.

## Contact

ProDiphy was developed by [Sebastian Proost](https://sebastian.proost.science/) at the 
[RaesLab](https://raeslab.sites.vib.be/en). ProDiphy is available under the 
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) 
license. 

For commercial access inquiries, please contact [Jeroen Raes](mailto:jeroen.raes@kuleuven.vib.be).
