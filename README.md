[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# ProDiphy

Python package that implements probabilistic models to compare (sub-)populations.

## Installation

Note: currently the package has not been deposited in a repository, so it needs to be installed through github.

To install Prodiphy, [conda](https://conda.io/projects/conda/en/latest/index.html) is required (or miniconda).

First, clone the repository to create a local copy in the current directory.

```bash
git clone https://github.com/raeslab/prodiphy
```

Then navigate to the source code, create an environment using the yaml file in the folder `docs/dev`, activate it and install the local copy of the code.

```bash
cd prodiphy
conda env create -f docs/devenvironment.yml
conda activate prodiphy
pip install -e .
```

## Usage

### ProDir Model

The ProDir model can be used to compare the prevalence of specific classes in two populations. E.g. two locations with
a similar ecosystem, one exposed to a pollutant and the other not. Counts, how many individuals per species where 
observed can be provided to the model, and the results will show if the model is confident there is a difference in 
prevalence of that species in the polluted vs unpolluted ecosystem.

```python
from prodiphy import ProDir

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
