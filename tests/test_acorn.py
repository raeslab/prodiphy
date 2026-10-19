import numpy as np
import pandas as pd
import pytest

from prodiphy import ACORN


@pytest.fixture(scope="module")
def sample_data():
    """
    Generate a small wide sample x feature table with a planted association.

    One feature ("feat_signal") is constructed to depend strongly on the marker,
    the others are noise. Covariates age (continuous) and sex (binary) are
    included so covariate handling is exercised.

    Returns:
        tuple: (DataFrame, list of count column names)
    """
    rng = np.random.default_rng(42)
    n_samples = 60

    marker = rng.normal(0, 1, n_samples)
    age = rng.normal(50, 10, n_samples)
    sex = rng.choice(["m", "f"], size=n_samples)

    feature_names = ["feat_signal", "feat_noise_1", "feat_noise_2", "feat_noise_3"]

    def nb_counts(log_mu, alpha=5.0):
        mu = np.exp(log_mu)
        # numpy parametrizes NB by (n successes, prob); convert from mu/alpha.
        p = alpha / (alpha + mu)
        return rng.negative_binomial(alpha, p)

    counts = {
        "feat_signal": nb_counts(3.0 + 0.8 * marker),
        "feat_noise_1": nb_counts(3.0 + 0.0 * marker),
        "feat_noise_2": nb_counts(2.5 + 0.0 * marker),
        "feat_noise_3": nb_counts(3.5 + 0.0 * marker),
    }

    df = pd.DataFrame({"marker": marker, "age": age, "sex": sex, **counts})
    return df, feature_names


@pytest.fixture(scope="module")
def fitted_model(sample_data):
    """Fit a small ACORN model (ADVI for speed) on the sample data."""
    df, feature_names = sample_data
    model = ACORN()
    model.fit(
        df,
        count_cols=feature_names,
        marker="marker",
        covariates=[("age", "continuous"), ("sex", {"m": 1, "f": 0})],
        method="advi",
    )
    return model


def test_init_with_default_parameters():
    model = ACORN()
    assert model.chains == 3
    assert model.cores == 4
    assert model.draws == 500
    assert model.tune == 500
    assert model.model is None
    assert model.trace is None


def test_init_with_custom_parameters():
    model = ACORN(chains=2, cores=2, draws=100, tune=100, random_seed=7)
    assert model.chains == 2
    assert model.cores == 2
    assert model.draws == 100
    assert model.tune == 100
    assert model.random_seed == 7


def test_fit_nuts(sample_data):
    df, feature_names = sample_data
    model = ACORN(chains=1, cores=1, draws=50, tune=50)
    model.fit(
        df,
        count_cols=feature_names,
        marker="marker",
        covariates=[("age", "continuous"), ("sex", {"m": 1, "f": 0})],
        method="nuts",
    )
    assert model.trace is not None
    assert model.model is not None
    assert "beta_marker" in model.trace.posterior


def test_fit_advi(fitted_model):
    assert fitted_model.trace is not None
    assert "beta_marker" in fitted_model.trace.posterior


def test_fit_without_covariates(sample_data):
    df, feature_names = sample_data
    model = ACORN()
    model.fit(df, count_cols=feature_names, marker="marker", method="advi")
    assert model.covariate_vars == []
    assert model.trace is not None


def test_fit_with_offset(sample_data):
    df, feature_names = sample_data
    model = ACORN()
    model.fit(
        df,
        count_cols=feature_names,
        marker="marker",
        offset=True,
        method="advi",
    )
    assert model.trace is not None


def test_fit_with_hierarchical_dispersion(sample_data):
    df, feature_names = sample_data
    model = ACORN()
    model.fit(
        df,
        count_cols=feature_names,
        marker="marker",
        hierarchical_dispersion=True,
        method="advi",
    )
    assert "dispersion" in model.trace.posterior
    assert "mu_dispersion" in model.trace.posterior


def test_fit_with_empty_dataframe():
    model = ACORN()
    with pytest.raises(ValueError):
        model.fit(pd.DataFrame(), count_cols=["a"], marker="marker", method="advi")


def test_fit_with_missing_column(sample_data):
    df, feature_names = sample_data
    model = ACORN()
    with pytest.raises(ValueError):
        model.fit(df, count_cols=feature_names, marker="not_a_column", method="advi")


def test_fit_with_no_count_cols(sample_data):
    df, _ = sample_data
    model = ACORN()
    with pytest.raises(ValueError):
        model.fit(df, count_cols=[], marker="marker", method="advi")


def test_fit_rejects_multilevel_categorical(sample_data):
    df, feature_names = sample_data
    df = df.copy()
    df["diet"] = np.resize(["a", "b", "c"], len(df))
    model = ACORN()
    with pytest.raises(ValueError):
        model.fit(
            df,
            count_cols=feature_names,
            marker="marker",
            covariates=[("diet", {"a": 0, "b": 1, "c": 2})],
            method="advi",
        )


def test_fit_rejects_unmapped_covariate_value(sample_data):
    df, feature_names = sample_data
    model = ACORN()
    with pytest.raises(ValueError):
        model.fit(
            df,
            count_cols=feature_names,
            marker="marker",
            # 'm' is present in the data but not in the encoding.
            covariates=[("sex", {"f": 0})],
            method="advi",
        )


def test_fit_invalid_method(sample_data):
    df, feature_names = sample_data
    model = ACORN()
    with pytest.raises(ValueError):
        model.fit(df, count_cols=feature_names, marker="marker", method="bogus")


def test_get_stats_returns_dataframe(fitted_model):
    table = fitted_model.get_stats()
    assert isinstance(table, pd.DataFrame)
    assert table.shape[0] == 4
    for col in [
        "feature",
        "log_effect",
        "hdi_94_low",
        "hdi_94_high",
        "fold_change",
        "prob_direction",
        "prob_outside_rope",
        "hdi_excludes_zero",
    ]:
        assert col in table.columns


def test_get_stats_custom_hdi(fitted_model):
    table = fitted_model.get_stats(hdi_prob=0.9)
    assert "hdi_90_low" in table.columns
    assert "hdi_90_high" in table.columns


def test_get_stats_recovers_signal(fitted_model):
    table = fitted_model.get_stats()
    # The planted feature should have the largest absolute log-effect.
    top = table.iloc[0]
    assert top["feature"] == "feat_signal"
    assert top["log_effect"] > 0


def test_get_stats_with_unfitted_model():
    model = ACORN()
    with pytest.raises(ValueError):
        model.get_stats()


def test_get_summary_returns_dataframe(fitted_model):
    summary = fitted_model.get_summary()
    assert isinstance(summary, pd.DataFrame)


def test_get_summary_with_unfitted_model():
    model = ACORN()
    with pytest.raises(ValueError):
        model.get_summary()


def test_fit_rejects_non_dataframe(sample_data):
    df, feature_names = sample_data
    model = ACORN()
    with pytest.raises(ValueError, match="DataFrame"):
        model.fit(
            df.to_numpy(), count_cols=feature_names, marker="marker", method="advi"
        )


def test_fit_rejects_constant_marker(sample_data):
    """A marker without variance cannot be z-scored, so the fit is refused."""
    df, feature_names = sample_data
    df = df.assign(marker=1.0)
    model = ACORN()
    with pytest.raises(ValueError, match="variance"):
        model.fit(df, count_cols=feature_names, marker="marker", method="advi")


def test_fit_rejects_unknown_encoding(sample_data):
    df, feature_names = sample_data
    model = ACORN()
    with pytest.raises(ValueError, match="must be 'continuous'"):
        model.fit(
            df,
            count_cols=feature_names,
            marker="marker",
            covariates=[("age", "categorical")],
            method="advi",
        )


def test_fit_with_no_complete_samples(sample_data):
    """Samples missing the marker are dropped; if none remain the fit is refused."""
    df, feature_names = sample_data
    df = df.assign(marker=np.nan)
    model = ACORN()
    with pytest.raises(ValueError, match="No samples remain"):
        model.fit(df, count_cols=feature_names, marker="marker", method="advi")


def test_fit_with_offset_column(sample_data):
    """
    An explicit offset column is used as log(total) per sample. With the per-sample totals in that column it matches
    the default offset computed from the count columns, and a different column changes the offset accordingly.
    """
    df, feature_names = sample_data
    totals = df[feature_names].sum(axis=1)
    df = df.assign(total=totals, double_total=2 * totals)

    model = ACORN()
    model.fit(
        df,
        count_cols=feature_names,
        marker="marker",
        offset=True,
        offset_col="total",
        method="advi",
    )
    assert model.trace is not None

    from_col = model._build_model_data(df, feature_names, "marker", [], True, "total")
    from_sum = model._build_model_data(df, feature_names, "marker", [], True, None)
    doubled = model._build_model_data(
        df, feature_names, "marker", [], True, "double_total"
    )
    np.testing.assert_allclose(from_col["offset"], from_sum["offset"])
    np.testing.assert_allclose(doubled["offset"] - from_col["offset"], np.log(2))


def test_fit_with_offset_column_missing(sample_data):
    df, feature_names = sample_data
    model = ACORN()
    with pytest.raises(ValueError):
        model.fit(
            df,
            count_cols=feature_names,
            marker="marker",
            offset=True,
            offset_col="total",
            method="advi",
        )


def test_fit_rejects_nonpositive_offset(sample_data):
    """A log(total) offset is undefined for samples with a total of zero."""
    df, feature_names = sample_data
    df = df.assign(total=df[feature_names].sum(axis=1))
    df.loc[0, "total"] = 0
    model = ACORN()
    with pytest.raises(ValueError, match="strictly positive"):
        model.fit(
            df,
            count_cols=feature_names,
            marker="marker",
            offset=True,
            offset_col="total",
            method="advi",
        )


def test_fit_fullrank_advi(sample_data):
    """The full-rank ADVI preview also recovers the planted marker association."""
    df, feature_names = sample_data
    model = ACORN(advi_iterations=10000, advi_draws=500)
    model.fit(df, count_cols=feature_names, marker="marker", method="fullrank_advi")

    table = model.get_stats()
    assert table.iloc[0]["feature"] == "feat_signal"
    assert table.iloc[0]["log_effect"] > 0
