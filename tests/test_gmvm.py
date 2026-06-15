import numpy as np
import pandas as pd
import pytest

from prodiphy import GMvM


@pytest.fixture(scope="module")
def sample_data():
    """
    Fixture to generate sample multivariate Gaussian data for testing.

    Returns:
        pd.DataFrame: A DataFrame containing the generated sample data.
    """
    # Generate 3 clusters of 2D data using numpy
    np.random.seed(42)

    # Cluster 1: centered at [0, 0]
    cluster1 = np.random.multivariate_normal([0, 0], [[0.8, 0], [0, 0.8]], 100)

    # Cluster 2: centered at [3, 3]
    cluster2 = np.random.multivariate_normal([3, 3], [[0.8, 0], [0, 0.8]], 100)

    # Cluster 3: centered at [-2, 2]
    cluster3 = np.random.multivariate_normal([-2, 2], [[0.8, 0], [0, 0.8]], 100)

    # Combine all clusters
    x = np.vstack([cluster1, cluster2, cluster3])

    # Convert to DataFrame
    df = pd.DataFrame(x, columns=["feature_1", "feature_2"])
    return df


@pytest.fixture(scope="module")
def high_dim_data():
    """
    Fixture to generate higher-dimensional sample data.

    Returns:
        pd.DataFrame: A DataFrame containing 4-dimensional sample data.
    """
    # Generate 3 clusters of 4D data (similar to Palmer penguins structure)
    np.random.seed(123)

    # Cluster 1: centered at [0, 0, 0, 0]
    cluster1 = np.random.multivariate_normal(
        [0, 0, 0, 0],
        [[0.6, 0, 0, 0], [0, 0.6, 0, 0], [0, 0, 0.6, 0], [0, 0, 0, 0.6]],
        50,
    )

    # Cluster 2: centered at [2, 2, 2, 2]
    cluster2 = np.random.multivariate_normal(
        [2, 2, 2, 2],
        [[0.6, 0, 0, 0], [0, 0.6, 0, 0], [0, 0, 0.6, 0], [0, 0, 0, 0.6]],
        50,
    )

    # Cluster 3: centered at [-1, 1, -1, 1]
    cluster3 = np.random.multivariate_normal(
        [-1, 1, -1, 1],
        [[0.6, 0, 0, 0], [0, 0.6, 0, 0], [0, 0, 0.6, 0], [0, 0, 0, 0.6]],
        50,
    )

    # Combine all clusters
    x = np.vstack([cluster1, cluster2, cluster3])

    df = pd.DataFrame(
        x, columns=["bill_length", "bill_depth", "flipper_length", "body_mass"]
    )
    return df


@pytest.fixture(scope="module")
def sample_model(sample_data):
    """
    Fixture to create and fit a GMvM model using the sample data.

    Args:
        sample_data (pd.DataFrame): The sample data to fit the model.

    Returns:
        GMvM: The fitted GMvM model.
    """
    model = GMvM(clusters=3, chains=2, cores=1, samples=50, tune=50)
    model.fit(sample_data)
    return model


def test_init_with_invalid_clusters():
    """
    Test initializing the GMvM model with invalid clusters parameter.
    """
    with pytest.raises(ValueError):
        GMvM(clusters=-1)


def test_init_with_default_parameters():
    """
    Test initializing the GMvM model with default parameters.
    """
    model = GMvM(clusters=3)
    assert model.clusters == 3
    assert model.chains == 4
    assert model.cores == 4
    assert model.samples == 1000
    assert model.tune == 1500
    assert model.model is None
    assert model.trace is None


def test_init_with_custom_parameters():
    """
    Test initializing the GMvM model with custom parameters.
    """
    model = GMvM(clusters=5, chains=2, cores=2, samples=500, tune=1000)
    assert model.clusters == 5
    assert model.chains == 2
    assert model.cores == 2
    assert model.samples == 500
    assert model.tune == 1000
    assert model.model is None
    assert model.trace is None


def test_fit_with_dataframe(sample_data):
    """
    Test fitting the GMvM model with a pandas DataFrame.

    Args:
        sample_data (pd.DataFrame): The sample data to fit the model.
    """
    model = GMvM(clusters=3, chains=2, cores=1, samples=50, tune=50)
    model.fit(sample_data)
    assert model.trace is not None
    assert model.model is not None
    assert model.obs is not None


def test_fit_with_numpy_array(sample_data):
    """
    Test fitting the GMvM model with a numpy array.

    Args:
        sample_data (pd.DataFrame): The sample data to convert and fit.
    """
    data_array = sample_data.values
    model = GMvM(clusters=3, chains=2, cores=1, samples=50, tune=50)
    model.fit(data_array)
    assert model.trace is not None
    assert model.model is not None


def test_fit_with_custom_parameters(sample_data):
    """
    Test fitting the GMvM model with custom prior parameters.

    Args:
        sample_data (pd.DataFrame): The sample data to fit the model.
    """
    model = GMvM(clusters=3, chains=1, cores=1, samples=50, tune=50)
    model.fit(sample_data, eta=3.0, sd=0.5, mu_prior_std=2.0)
    assert model.trace is not None
    assert model.model is not None


def test_fit_with_high_dimensional_data(high_dim_data):
    """
    Test fitting the GMvM model with higher-dimensional data.

    Args:
        high_dim_data (pd.DataFrame): The high-dimensional sample data.
    """
    model = GMvM(clusters=3, chains=2, cores=1, samples=50, tune=50)
    model.fit(high_dim_data)
    assert model.trace is not None
    assert model.model is not None


def test_fit_with_empty_dataframe():
    """
    Test fitting the GMvM model with an empty DataFrame.
    """
    empty_df = pd.DataFrame()
    model = GMvM(clusters=3, chains=1, cores=1, samples=50, tune=50)
    with pytest.raises(ValueError):
        model.fit(empty_df)


def test_fit_with_empty_array():
    """
    Test fitting the GMvM model with an empty numpy array.
    """
    empty_array = np.array([])
    model = GMvM(clusters=3, chains=1, cores=1, samples=50, tune=50)
    with pytest.raises(ValueError):
        model.fit(empty_array)


def test_fit_with_insufficient_observations():
    """
    Test fitting the GMvM model with fewer observations than clusters.
    """
    # Create data with only 2 observations but 3 clusters
    small_data = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "y"])
    model = GMvM(clusters=3, chains=1, cores=1, samples=50, tune=50)
    with pytest.raises(ValueError):
        model.fit(small_data)


def test_fit_with_1d_data():
    """
    Test fitting the GMvM model with 1-dimensional data (should fail).
    """
    data_1d = np.array([1, 2, 3, 4, 5])
    model = GMvM(clusters=2, chains=1, cores=1, samples=50, tune=50)
    with pytest.raises(ValueError):
        model.fit(data_1d)


def test_get_stats_returns_dataframe(sample_model):
    """
    Test that the get_stats method returns a DataFrame.

    Args:
        sample_model (GMvM): The fitted GMvM model.
    """
    stats_df = sample_model.get_stats()
    assert isinstance(stats_df, pd.DataFrame)


def test_get_stats_with_unfitted_model():
    """
    Test that the get_stats method raises an error when called on an unfitted model.
    """
    model = GMvM(clusters=3)
    with pytest.raises(ValueError):
        model.get_stats()


def test_get_clusters_with_unfitted_model(sample_data):
    """
    Test get_clusters method with an unfitted model.
    """
    model = GMvM(clusters=3)
    with pytest.raises(ValueError):
        model.get_clusters(sample_data)


def test_get_clusters_returns_dataframe(sample_data, sample_model):
    """
    Test that the get_clusters method returns a DataFrame with correct structure.
    """
    cluster_df = sample_model.get_clusters(sample_data)
    assert isinstance(cluster_df, pd.DataFrame)
    assert cluster_df.shape[0] == sample_data.shape[0]
    assert "C1" in cluster_df.columns
    assert "C2" in cluster_df.columns
    assert "C3" in cluster_df.columns
    assert "cluster" in cluster_df.columns


def test_get_clusters_with_different_data(sample_model):
    """
    Test get_clusters method with different data than what was used for fitting.
    """
    # Create new test data
    new_data = pd.DataFrame(
        [[0, 0], [3, 3], [-2, 2]], columns=["feature_1", "feature_2"]
    )
    cluster_df = sample_model.get_clusters(new_data)
    assert isinstance(cluster_df, pd.DataFrame)
    assert cluster_df.shape[0] == new_data.shape[0]


def test_determine_best_cluster_count(sample_data):
    """
    Test determine_best_cluster_count method with default parameters.
    """
    comps = GMvM.determine_best_cluster_count(
        sample_data, cluster_sizes=[2, 3], samples=50, tune=50, chains=1, cores=1
    )
    assert isinstance(comps, pd.DataFrame)
    assert comps.shape[0] == 2
    assert "rank" in comps.columns
    assert "p_waic" in comps.columns
    assert "elpd_waic" in comps.columns


def test_determine_best_cluster_count_with_loo(sample_data):
    """
    Test determine_best_cluster_count method with LOO criterion.
    """
    comps = GMvM.determine_best_cluster_count(
        sample_data,
        cluster_sizes=[2, 3],
        samples=50,
        tune=50,
        chains=1,
        cores=1,
        ic="loo",
    )
    assert isinstance(comps, pd.DataFrame)
    assert comps.shape[0] == 2
    assert "rank" in comps.columns
    assert "p_loo" in comps.columns
    assert "elpd_loo" in comps.columns


def test_determine_best_cluster_count_with_custom_priors(sample_data):
    """
    Test determine_best_cluster_count method with custom prior parameters.
    """
    comps = GMvM.determine_best_cluster_count(
        sample_data,
        cluster_sizes=[2, 3],
        samples=50,
        tune=50,
        chains=1,
        cores=1,
        eta=3.0,
        sd=0.5,
        mu_prior_std=2.0,
    )
    assert isinstance(comps, pd.DataFrame)
    assert comps.shape[0] == 2
