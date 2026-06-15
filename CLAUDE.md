# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProDiphy is a Python package that implements probabilistic models to compare (sub-)populations using Bayesian statistics. It provides six main models:

- **ProDir**: Compare prevalence of specific classes in two populations using Dirichlet distributions
- **CorProDir**: ProDir model with covariate correction using Bambi/hierarchical modeling
- **DeltaSlope**: Compare slope, intercept and spread of linear regressions between two groups
- **DMM**: Dirichlet Multinomial Mixture model to detect clusters with different prevalence patterns
- **GMvM**: Gaussian Multivariate Mixture model for clustering continuous multivariate data
- **ACORN**: Adjusted CORrelations, Negative-binomial; hierarchical NB model relating every feature in a count table to a continuous marker, adjusted for covariates

## Configuration

This project uses modern Python packaging standards with all configuration centralized in `pyproject.toml`:
- **Package metadata**: Dependencies, version, authors
- **Build system**: PEP 517/518 compliant (setuptools backend)
- **Testing**: pytest configuration
- **Linting/Formatting**: ruff configuration
- **Coverage**: coverage.py configuration

## Development Commands

### Installation and Environment Setup
```bash
# Create conda environment and install package
conda create -n prodiphy python=3.12
conda activate prodiphy
pip install -e .
```

### Testing
```bash
# Run all tests with coverage
pytest --exitfirst --verbose --failed-first --cov=src tests/ --cov-report=term-missing --cov-report=xml

# Run specific test file
pytest tests/test_prodir.py -v

# Run tests for a specific model
pytest -k "prodir" -v
```

### Code Formatting and Linting
```bash
# Install ruff (if not already installed)
pip install ruff

# Check code formatting
ruff format --check .

# Format code
ruff format .

# Lint code (check for issues)
ruff check .

# Lint and auto-fix issues
ruff check --fix .
```

## Code Architecture

### Core Structure
- `src/prodiphy/` - Main package containing all model implementations
- `tests/` - Test suite with one test file per model
- `docs/` - Model documentation and development environment files
- `.github/workflows/` - CI/CD with automated testing and code formatting

### Model Implementation Pattern
All models follow a consistent pattern:
1. **Initialization**: Set up chains, cores, and sampling parameters
2. **fit()**: Main method that defines the Bayesian model using PyMC and fits it to data
3. **get_stats()**: Extract and format results from the fitted model trace

### Key Dependencies
- **PyMC**: Core Bayesian modeling framework (≥5.16.2)
- **Bambi**: High-level interface for hierarchical models (≥0.14.1)
- **ArviZ**: Bayesian analysis and visualization (≥0.19.0)
- **NumPy/Pandas**: Data manipulation
- **SciPy**: Statistical functions

### Model-Specific Details

**ProDir** (`prodir.py`): Simplest model comparing two groups using Dirichlet distributions
- Input: Two lists of counts and labels
- Output: Group proportions, deltas, and log2 ratios with credible intervals

**CorProDir** (`corprodir.py`): Most complex model using Bambi for hierarchical modeling with covariates
- Uses formula syntax for model specification
- Supports both corrected and uncorrected model fitting
- Handles categorical and continuous covariates

**DeltaSlope** (`deltaslope.py`): Linear regression comparison between two groups
- Input: Two DataFrames with x,y data
- Fits separate linear models per group and compares parameters

**DMM** (`dmm.py`): Mixture model for cluster detection
- Uses Dirichlet-Multinomial distributions
- Automatically determines optimal number of clusters

**GMvM** (`gmvm.py`): Gaussian Multivariate Mixture model for continuous data clustering
- Input: Continuous multivariate data (DataFrames or numpy arrays)
- Uses separate covariance matrices per cluster with LKJ priors
- Output: Cluster assignments with probabilities and model parameters
- Includes model comparison functionality for optimal cluster selection

**ACORN** (`acorn.py`): Adjusted CORrelations, Negative-binomial; hierarchical NB correlation model
- Input: Wide DataFrame (one row per sample) with feature count columns, a continuous marker column, and covariate columns
- Fits all features jointly; each per-feature coefficient is non-centered and partially pooled (principled alternative to per-feature GLM + FDR)
- Covariates are explicit `(column, encoding)` specs: `"continuous"` (z-scored) or a `{value: 0/1}` dict (binary); multi-level categoricals are rejected
- Inference via `method="nuts"` (default `nuts_sampler="pymc"`; pass `"numpyro"` if installed) or `method="advi"`/`"fullrank_advi"` for fast previews
- Optional `log(total)` offset for non-depth-normalized tables and optional hierarchical pooling of the dispersion
- `get_stats()` returns the ranked association table (log_effect, HDI, fold_change, prob_direction, prob_outside_rope, hdi_excludes_zero); `get_summary()` returns a raw ArviZ summary

## Development Guidelines

### Testing Strategy
- Each model has comprehensive tests in `tests/test_<model>.py`
- Tests cover basic functionality, edge cases, and output format validation
- CI runs tests on Python 3.10, 3.11, and 3.12

### Code Quality
- Ruff formatting and linting enforced in CI
- Code must pass `ruff format --check .` and `ruff check .` before merging
- Coverage reporting generated and tracked via badges

### Model Development
- All models inherit sampling parameters: `chains`, `cores`, `draws`, `tune`
- Use `sample_kwargs` parameter to pass additional arguments to `pm.sample()`
- Follow existing patterns for credible interval extraction and result formatting
- Include both point estimates and uncertainty quantification in outputs