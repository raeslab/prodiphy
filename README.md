[![Run Pytest](https://github.com/raeslab/prodiphy/actions/workflows/autopytest.yml/badge.svg)](https://github.com/raeslab/prodiphy/actions/workflows/autopytest.yml) [![Coverage](https://raw.githubusercontent.com/raeslab/prodiphy/main/docs/coverage-badge.svg)](https://raw.githubusercontent.com/raeslab/prodiphy/main/docs/coverage-badge.svg) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# ProDiphy

Python package that implements probabilistic models to compare (sub-)populations.

## Installation

Note: currently the package has not been deposited in a repository, so it needs to be installed through GitHub.

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
pip install -e .
```

## Quick Start

The following code snippet shows how to use the ProDir model to compare two populations. The model is used here to compare 
the prevalence of species in two ecosystems, one polluted and one unpolluted. The model will estimate the 
prevalence of each species in both ecosystems and provide a confidence interval for the difference in prevalence.

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

For more details including the output of this example, see the [ProDir documentation](./docs/prodir).

## Usage

The Prodiphy package contains various models each with their own specific use case. The following models are currently
available:

  * [ProDir](./docs/prodir.md): Model to compare the prevalence of specific classes in two populations.
  * [CorProDir](./docs/corprodir.md): Model to compare the prevalence of specific classes in two populations while correcting for covariates.
  * [DeltaSlope](./docs/deltaslope.md): Model to compare the slope, intercept and spread of a linear regression between two groups.
  * [DMM](./docs/dmm.md): Dirichlet Multinomial Mixture model to detect clusters with different prevalence of classes.


## Contributing

Any contributions you make are **greatly appreciated**.

  * Found a bug or have some suggestions? Open an [issue](https://github.com/raeslab/prodiphy/issues).
  * Pull requests are welcome! Though open an [issue](https://github.com/raeslab/prodiphy/issues) first to discuss which features/changes you wish to implement.

## Contact

ProDiphy was developed by [Sebastian Proost](https://sebastian.proost.science/) at the 
[RaesLab](https://raeslab.sites.vib.be/en). ProDiphy is available under the 
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) 
license. 

For commercial access inquiries, please contact [Jeroen Raes](mailto:jeroen.raes@kuleuven.vib.be).
