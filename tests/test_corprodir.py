import numpy as np
import pandas as pd
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


@pytest.fixture
def sample_data_large():
    labels = ["a", "b", "c", "d"]
    ref_size = 100
    target_size = 1100  # Needs to be > 1000
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
    """
    Test custom initialization values for the CorProDir class.

    This test verifies that the CorProDir class can be initialized with custom values
    for the number of chains, cores, and draws. It checks that the attributes of the
    instance match the provided custom values.
    """
    model = CorProDir(chains=3, cores=2, draws=500)
    assert model.chains == 3
    assert model.cores == 2
    assert model.draws == 500


def test_corprodir(sample_data):
    """
    Test the CorProDir class with sample data.

    This test verifies the functionality of the CorProDir class by fitting it with
    reference and target data, and then checking the resulting statistics DataFrame.

    Args:
        sample_data (tuple): A tuple containing reference DataFrame, target DataFrame, and labels list.
    """
    n_draws = 200
    ref_df, target_df, labels = sample_data

    model = CorProDir(draws=n_draws, chains=2, cores=2)

    model.fit(ref_df, target_df, "label", ["age", "BMI"])

    stats_df = model.get_stats()

    assert isinstance(stats_df, pd.DataFrame)
    assert stats_df.shape[0] == len(
        labels
    )  # should match the number of labels in the dataset
    assert stats_df.shape[1] == 20

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
        "mean_log_odds_ratio",
        "std_log_odds_ratio",
        "hdi_low_log_odds_ratio",
        "hdi_high_log_odds_ratio",
        "odds_ratio",
        "hdi_low_odds_ratio",
        "hdi_high_odds_ratio",
    ]
    for col in expected_columns:
        assert col in stats_df.columns


def test_corprodir_large(sample_data_large):
    """
    Test the CorProDir class with sample data large.

    This test emulates an edge case, but it doesn't seem to matter anymore since the update in bambi (will be removed
    once confirmed).
    """
    n_draws = 100
    ref_df, target_df, labels = sample_data_large

    model = CorProDir(draws=n_draws, chains=2, cores=2)

    model.fit(ref_df, target_df, "label", ["age", "BMI"])

    stats_df = model.get_stats()

    assert isinstance(stats_df, pd.DataFrame)


def test_corprodir_invalid_kind(sample_data):
    """An unknown prediction kind is rejected before any sampling happens."""
    ref_df, target_df, _ = sample_data

    model = CorProDir(draws=100, chains=2, cores=2)

    with pytest.raises(ValueError):
        model.fit(ref_df, target_df, "label", ["age", "BMI"], kind="mean")


def test_corprodir_does_not_mutate_input(sample_data):
    """Fitting does not add the one-hot encoded columns to the caller's DataFrames."""
    ref_df, target_df, _ = sample_data
    ref_columns = ref_df.columns.tolist()
    target_columns = target_df.columns.tolist()

    model = CorProDir(draws=100, chains=2, cores=2)
    model.fit(ref_df, target_df, "label", ["age", "BMI"])

    assert ref_df.columns.tolist() == ref_columns
    assert target_df.columns.tolist() == target_columns


def test_corprodir_odds_ratio_recovery():
    """
    The one-vs-rest odds ratio (target vs expected) recovers the known value on synthetic data where the
    confounders are unrelated to the label.
    """
    rng = np.random.default_rng(42)
    labels = ["a", "b", "c", "d"]
    ref_size = 2000
    target_size = 2000
    ref_prevalence = [0.25, 0.25, 0.25, 0.25]
    target_prevalence = [0.4, 0.2, 0.2, 0.2]

    ref_df = pd.DataFrame(
        {
            "age": rng.normal(size=ref_size),
            "label": rng.choice(labels, size=ref_size, p=ref_prevalence),
        }
    )
    target_df = pd.DataFrame(
        {
            "age": rng.normal(size=target_size),
            "label": rng.choice(labels, size=target_size, p=target_prevalence),
        }
    )

    model = CorProDir(draws=300, chains=2, cores=2)
    model.fit(ref_df, target_df, "label", ["age"])

    stats_df = model.get_stats().set_index("label")

    expected = {"a": (0.4 / 0.6) / (0.25 / 0.75), "b": (0.2 / 0.8) / (0.25 / 0.75)}
    for label, odds_ratio in expected.items():
        assert stats_df.loc[label, "hdi_low_odds_ratio"] < odds_ratio
        assert stats_df.loc[label, "hdi_high_odds_ratio"] > odds_ratio
    assert stats_df.loc["a", "hdi_low_odds_ratio"] > 1
    assert stats_df.loc["b", "hdi_high_odds_ratio"] < 1


def test_corprodir_missing_category_in_target():
    """
    A label present in the reference but absent from a small target gives finite summaries with
    kind="response_params" (the default).
    """
    rng = np.random.default_rng(7)
    labels = ["a", "b", "c", "d"]
    ref_size = 400
    target_size = 30

    ref_df = pd.DataFrame(
        {
            "age": rng.normal(size=ref_size),
            "label": rng.choice(labels, size=ref_size, p=[0.4, 0.3, 0.25, 0.05]),
        }
    )
    target_df = pd.DataFrame(
        {
            "age": rng.normal(size=target_size),
            "label": rng.choice(labels[:3], size=target_size, p=[0.4, 0.3, 0.3]),
        }
    )

    model = CorProDir(draws=200, chains=2, cores=2)
    model.fit(ref_df, target_df, "label", ["age"])

    stats_df = model.get_stats().set_index("label")

    assert stats_df.index.tolist() == labels
    assert np.isfinite(stats_df.to_numpy(dtype=float)).all()
    assert stats_df.loc["d", "mean_fraction"] < 0.1


def test_corprodir_kind_response(sample_data):
    """
    With kind="response" the expected prevalence is the fraction of simulated outcomes, so every draw is a
    multiple of 1 / n_target. With the default kind="response_params" it is an average of probabilities.
    """
    ref_df, target_df, labels = sample_data
    n_target = len(target_df)

    model = CorProDir(draws=100, chains=2, cores=2)
    model.fit(ref_df, target_df, "label", ["age", "BMI"], kind="response")

    assert model.kind == "response"
    assert model.get_stats().shape == (len(labels), 20)
    for label in labels:
        scaled = model.final_data[f"{label}_prevalence_est"].to_numpy() * n_target
        np.testing.assert_allclose(scaled, np.round(scaled), atol=1e-9)

    model.fit(ref_df, target_df, "label", ["age", "BMI"])

    assert model.kind == "response_params"
    scaled = model.final_data[f"{labels[0]}_prevalence_est"].to_numpy() * n_target
    assert not np.allclose(scaled, np.round(scaled))
