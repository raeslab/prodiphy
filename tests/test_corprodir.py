import pandas as pd
import numpy as np
import pytest

from prodiphy import CorProDir

np.random.seed(1910)


@pytest.fixture
def sample_data():
    labels = ["a", "b", "c", "d"]
    ref_size = 100
    target_size = 50
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

    return ref_df, target_df, labels


def test_init_defaults():
    """Test default initialization of the CorProDir class."""
    model = CorProDir()
    assert model.chains == 4
    assert model.cores == 4
    assert model.draws == 1000
    assert model.model is None
    assert model.trace is None


def test_init_custom():
    """Test custom initialization values."""
    model = CorProDir(chains=3, cores=2, draws=500)
    assert model.chains == 3
    assert model.cores == 2
    assert model.draws == 500


def test_corprodir(sample_data):
    n_draws = 200
    ref_df, target_df, labels = sample_data

    model = CorProDir(draws=n_draws)

    stats_df = model.fit(ref_df, target_df, "label", ["age", "BMI"])

    assert isinstance(stats_df, pd.DataFrame)
    assert stats_df.shape[0] == len(
        labels
    )  # should match the number of label in the dataset
    assert stats_df.shape[1] == 13

    for label in labels:
        assert label in stats_df["label"].tolist()

    expected_columns = [
        "label",
        "mean_fraction",
        "mean_estimate",
        "mean_delta",
        "std_delta",
        "hdi_low_delta",
        "hdi_high_delta",
        "mean_log2_ratio",
        "std_log2_ratio",
        "hdi_low_log2_ratio",
        "hdi_high_log2_ratio",
        "fraction_above_zero",
        "fraction_below_zero",
    ]
    for col in expected_columns:
        assert col in stats_df.columns
