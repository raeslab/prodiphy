import os

import numpy as np
import pytest

from prodiphy import ProDir

# Set PyTensor flags to suppress warning
os.environ["PYTENSOR_FLAGS"] = "cxx="


# @pytest.mark.skip(reason="Temporarily disabled to speed up testing of other components")
def test_prodir():
    model = ProDir()

    # Input data for group 1, group 2, and their corresponding labels
    group_1_counts = [200, 300, 150, 100]
    group_2_counts = [10, 33, 12, 8]
    sub_group_labels = ["a", "b", "c", "d"]

    # Fit the model with input data and enable verbose output
    model.fit(group_1_counts, group_2_counts, sub_group_labels, verbose=True)

    # Generate a summary of the model trace using Arviz
    summary_df = model.get_stats()

    # Check that 'delta' and 'log2_ratio' variables are present in the summary for each label
    for label in sub_group_labels:
        assert f"delta_{label}" in summary_df.index
        assert f"log2_ratio_{label}" in summary_df.index

    # Check that the proportions in group_1 match the expected values
    for v, label in zip(group_1_counts, sub_group_labels, strict=True):
        proportion_obs = v / sum(group_1_counts)
        assert f"group_1_p_{label}" in summary_df.index
        assert summary_df.loc[f"group_1_p_{label}", "mean"] == pytest.approx(
            proportion_obs, rel=1e-1
        )

    # Check that the proportions in group_2 match the expected values
    for v, label in zip(group_2_counts, sub_group_labels, strict=True):
        proportion_obs = v / sum(group_2_counts)
        assert f"group_2_p_{label}" in summary_df.index
        assert summary_df.loc[f"group_2_p_{label}", "mean"] == pytest.approx(
            proportion_obs, rel=1e-1
        )

    # Ensure an exception is raised if input lists have different lengths
    with pytest.raises(AssertionError):
        model.fit(group_1_counts, group_2_counts, sub_group_labels[2:], verbose=True)
    with pytest.raises(AssertionError):
        model.fit(group_1_counts, group_2_counts[2:], sub_group_labels, verbose=True)


def test_prodir_odds_ratio_recovery():
    """The one-vs-rest odds ratio (group 2 vs group 1) recovers the known value on synthetic counts."""
    model = ProDir(chains=2, cores=2)

    # p1 = 0.25 for every label, p2 = [0.4, 0.2, 0.2, 0.2]
    group_1_counts = [2500, 2500, 2500, 2500]
    group_2_counts = [4000, 2000, 2000, 2000]
    labels = ["a", "b", "c", "d"]
    model.fit(group_1_counts, group_2_counts, labels)

    summary_df = model.get_stats()

    expected = {"a": (0.4 / 0.6) / (0.25 / 0.75), "b": (0.2 / 0.8) / (0.25 / 0.75)}
    for label, odds_ratio in expected.items():
        assert summary_df.loc[f"log_odds_ratio_{label}", "mean"] == pytest.approx(
            np.log(odds_ratio), abs=0.05
        )
        assert summary_df.loc[f"odds_ratio_{label}", "mean"] == pytest.approx(
            odds_ratio, rel=0.05
        )


def test_prodir_odds_ratio_zero_count():
    """A label with zero counts in one group yields a finite odds ratio, clearly below one."""
    model = ProDir(chains=2, cores=2)

    group_1_counts = [100, 100, 100, 100]
    group_2_counts = [0, 150, 150, 100]
    labels = ["a", "b", "c", "d"]
    model.fit(group_1_counts, group_2_counts, labels)

    summary_df = model.get_stats()

    assert (
        np.isfinite(summary_df.loc[:, ["mean", "sd", "hdi_3%", "hdi_97%"]]).all().all()
    )
    assert summary_df.loc["odds_ratio_a", "hdi_97%"] < 1
    assert summary_df.loc["log_odds_ratio_a", "hdi_97%"] < 0
