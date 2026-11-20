import os

import numpy as np
import pandas as pd
import pytest

from prodiphy import DeltaSlopeHierarchical

# Set PyTensor flags to suppress warning
os.environ["PYTENSOR_FLAGS"] = "cxx="


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    x_ref = rng.integers(0, 100, size=200)
    y_ref = x_ref * 2 + 10 + rng.normal(0, 2, size=200)

    x_target = rng.integers(0, 100, size=200)
    y_target = x_target * 1.5 + 9 + rng.normal(0, 3, size=200)

    ref_df = pd.DataFrame({"x": x_ref, "y": y_ref})
    target_df = pd.DataFrame({"x": x_target, "y": y_target})

    return ref_df, target_df


def test_deltaslope_hierarchical_init():
    model = DeltaSlopeHierarchical(
        draws=1000,
        tune=500,
        chains=2,
        cores=1,
        sigma_slope_prior=0.2,
        sigma_intercept_prior=0.3,
    )

    assert model.draws == 1000
    assert model.tune == 500
    assert model.chains == 2
    assert model.cores == 1
    assert model.sigma_slope_prior == 0.2
    assert model.sigma_intercept_prior == 0.3
    assert model.model is None
    assert model.trace is None


def test_deltaslope_hierarchical_fit(sample_data):
    ref_df, target_df = sample_data
    model = DeltaSlopeHierarchical(draws=100, tune=50, chains=1, cores=1)
    model.fit(ref_df, target_df, x="x", y="y", sample_kwargs={"random_seed": 1})

    assert model.model is not None
    assert model.trace is not None
    assert len(model.trace.posterior) > 0
    # Inputs must not be modified
    assert list(ref_df.columns) == ["x", "y"]
    assert list(target_df.columns) == ["x", "y"]


def test_deltaslope_hierarchical_stats(sample_data):
    ref_df, target_df = sample_data
    model = DeltaSlopeHierarchical(draws=300, tune=300, chains=2, cores=1)
    model.fit(ref_df, target_df, x="x", y="y", sample_kwargs={"random_seed": 1})

    summary_df = model.get_stats()

    assert isinstance(summary_df, pd.DataFrame)
    assert "mean" in summary_df.columns
    assert "hdi_3%" in summary_df.columns
    assert set(summary_df.index) == {
        f"{p}_{v}"
        for p in ["delta", "ref", "target"]
        for v in ["at_mean_x", "intercept", "sigma", "slope"]
    }

    # Back-transformed parameters should recover the simulated values
    means = summary_df["mean"]
    assert means["ref_slope"] == pytest.approx(2.0, abs=0.1)
    assert means["target_slope"] == pytest.approx(1.5, abs=0.1)
    assert means["delta_slope"] < 0
    assert means["ref_sigma"] == pytest.approx(2.0, abs=0.5)
    assert means["target_sigma"] == pytest.approx(3.0, abs=0.6)
    assert means["delta_slope"] == pytest.approx(
        means["target_slope"] - means["ref_slope"], abs=1e-6
    )


def test_deltaslope_hierarchical_invalid_input():
    ref_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    target_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    model = DeltaSlopeHierarchical(draws=100, tune=50, chains=1, cores=1)

    with pytest.raises(KeyError):
        model.fit(ref_df, target_df, x="invalid_column", y="b")


def test_deltaslope_hierarchical_constant_column():
    ref_df = pd.DataFrame({"a": [1, 1, 1], "b": [4, 5, 6]})
    target_df = pd.DataFrame({"a": [1, 1, 1], "b": [4, 5, 6]})

    model = DeltaSlopeHierarchical(draws=100, tune=50, chains=1, cores=1)

    with pytest.raises(ValueError, match="must both vary"):
        model.fit(ref_df, target_df, x="a", y="b")
