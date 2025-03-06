import arviz as az
import pytest
import os
import numpy as np
import pandas as pd

from prodiphy import DeltaSlope

# Set PyTensor flags to suppress warning
os.environ["PYTENSOR_FLAGS"] = "cxx="

@pytest.fixture
def sample_data():
    x_ref = np.random.randint(0, 100, size=200)
    y_ref = x_ref * 2 + 10 + np.random.normal(0,2,size=200)

    x_target = np.random.randint(0, 100, size=200)
    y_target = x_target * 1.5 + 9 + np.random.normal(0,3,size=200)

    ref_df = pd.DataFrame({"x": x_ref, "y": y_ref})
    target_df = pd.DataFrame({"x": x_target, "y": y_target})

    return ref_df, target_df


def test_deltaslope_init():
    """Test initialization of the DeltaSlope class."""
    model = DeltaSlope(draws=1000, tune=500, chains=2, cores=1)

    assert model.draws == 1000
    assert model.tune == 500
    assert model.chains == 2
    assert model.cores == 1
    assert model.model is None
    assert model.trace is None


def test_deltaslope_fit(sample_data):
    ref_df, target_df = sample_data
    model = DeltaSlope(draws=100, tune=50, chains=1, cores=1)  # smaller values for faster test
    model.fit(ref_df, target_df, x="x", y="y")

    assert model.model is not None
    assert model.trace is not None
    assert len(model.trace.posterior) > 0


def test_deltaslope_stats(sample_data):
    ref_df, target_df = sample_data
    model = DeltaSlope(draws=100, tune=50, chains=2, cores=1)  # Two chains are needed for the summary stats to work
    model.fit(ref_df, target_df, x="x", y="y")

    summary_df = model.get_stats()

    assert not summary_df.empty
    assert isinstance(summary_df, pd.DataFrame)
    assert "mean" in summary_df.columns
    assert "hdi_3%" in summary_df.columns


def test_deltaslope_invalid_input():
    """Test the fit method with invalid input."""
    ref_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    target_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    model = DeltaSlope(draws=100, tune=50, chains=1, cores=1)

    with pytest.raises(KeyError):
        model.fit(ref_df, target_df, x="invalid_column", y="b")