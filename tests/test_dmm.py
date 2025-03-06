import pytest
import numpy as np
import pandas as pd
from random import shuffle
from scipy.stats import dirichlet

from prodiphy import DMM

@pytest.fixture
def sample_data():
    """
    Fixture to generate sample data for testing.

    Returns:
        pd.DataFrame: A DataFrame containing the generated sample data.
    """
    data = []

    alphas = [[16,1,1,1,1],
              [1,4,4,10,1],
              [2,2,2,2,20]]

    sample_counts = [100,100,200]

    for sample_count, alpha in zip(sample_counts, alphas):
        for j in range(sample_count):
            pvals = dirichlet.rvs(alpha, size=1)[0]
            data.append(np.random.multinomial(1000, pvals))

    shuffle(data)

    df = pd.DataFrame(data)
    return df

@pytest.fixture
def sample_model(sample_data):
    """
    Fixture to create and fit a DMM model using the sample data.

    Args:
        sample_data (pd.DataFrame): The sample data to fit the model.

    Returns:
        DMM: The fitted DMM model.
    """
    model = DMM(clusters=3, chains=2, cores=1, samples=5, tune=10)
    model.fit(sample_data)
    return model

def test_init_with_default_parameters():
    """
    Test initializing the DMM model with default parameters.
    """
    model = DMM(clusters=3)
    assert model.clusters == 3
    assert model.chains == 4
    assert model.cores == 4
    assert model.samples == 1000
    assert model.tune == 1500
    assert model.model is None
    assert model.trace is None

def test_init_with_custom_parameters():
    """
    Test initializing the DMM model with custom parameters.
    """
    model = DMM(clusters=5, chains=2, cores=2, samples=500, tune=1000)
    assert model.clusters == 5
    assert model.chains == 2
    assert model.cores == 2
    assert model.samples == 500
    assert model.tune == 1000
    assert model.model is None
    assert model.trace is None

def test_fit_with_default_parameters(sample_data):
    """
    Test fitting the DMM model with default parameters.

    Args:
        sample_data (pd.DataFrame): The sample data to fit the model.
    """
    ref_df = sample_data
    model = DMM(clusters=3, chains=2, cores=1, samples=5, tune=10)
    model.fit(ref_df)
    assert model.trace is not None
    assert model.model is not None

def test_fit_with_custom_parameters(sample_data):
    """
    Test fitting the DMM model with custom parameters.

    Args:
        sample_data (pd.DataFrame): The sample data to fit the model.
    """
    ref_df = sample_data
    model = DMM(clusters=3, chains=1, cores=1, samples=5, tune=10)
    model.fit(ref_df, lower=10, upper=100)
    assert model.trace is not None
    assert model.model is not None

def test_fit_with_priors_and_weights(sample_data):
    """
    Test fitting the DMM model with priors and weights.

    Args:
        sample_data (pd.DataFrame): The sample data to fit the model.
    """
    ref_df = sample_data
    priors = np.ones((3, ref_df.shape[1])) / ref_df.shape[1]
    weights = np.array([0.2, 0.5, 0.3])
    model = DMM(clusters=3, chains=1, cores=1, samples=5, tune=10)
    model.fit(ref_df, priors=priors, weights=weights)
    assert model.trace is not None
    assert model.model is not None

def test_fit_with_empty_dataframe():
    """
    Test fitting the DMM model with an empty DataFrame.
    """
    empty_df = pd.DataFrame()
    model = DMM(clusters=3, chains=1, cores=1, samples=5, tune=10)
    with pytest.raises(ValueError):
        model.fit(empty_df)

def test_get_stats_returns_dataframe(sample_model):
    """
    Test that the get_stats method returns a DataFrame.

    Args:
        sample_model (DMM): The fitted DMM model.
    """
    stats_df = sample_model.get_stats()
    assert isinstance(stats_df, pd.DataFrame)

def test_get_stats_with_unfitted_model():
    """
    Test that the get_stats method raises an error when called on an unfitted model.
    """
    model = DMM(clusters=3)
    with pytest.raises(ValueError):
        model.get_stats()