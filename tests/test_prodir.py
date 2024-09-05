import arviz as az
import pytest
import os

from prodiphy import ProDir

# Set PyTensor flags to suppress warning
os.environ["PYTENSOR_FLAGS"] = "cxx="

def test_prodir():
    model = ProDir()

    # Input data for group 1, group 2, and their corresponding labels
    group_1_counts = [200, 300, 150, 100]
    group_2_counts = [10, 33, 12, 8]
    sub_group_labels = ["a", "b", "c", "d"]

    # Fit the model with input data and enable verbose output
    model.fit(group_1_counts, group_2_counts, sub_group_labels, verbose=True)

    # Generate a summary of the model trace using Arviz
    with model.model:
        summary_df = az.summary(model.trace)

    # Check that 'delta' and 'log2_ratio' variables are present in the summary for each label
    for l in sub_group_labels:
        assert f"delta_{l}" in summary_df.index
        assert f"log2_ratio_{l}" in summary_df.index

    # Check that the proportions in group_1 match the expected values
    for ix, v in enumerate(group_1_counts):
        proportion_obs = v/sum(group_1_counts)
        assert f"group_1_p[{ix}]" in summary_df.index
        assert summary_df.loc[f"group_1_p[{ix}]", "mean"] == pytest.approx(proportion_obs, rel=1e-1)

    # Check that the proportions in group_2 match the expected values
    for ix, v in enumerate(group_2_counts):
        proportion_obs = v/sum(group_2_counts)
        assert f"group_2_p[{ix}]" in summary_df.index
        assert summary_df.loc[f"group_2_p[{ix}]", "mean"] == pytest.approx(proportion_obs, rel=1e-1)

    # Ensure an exception is raised if input lists have different lengths
    with pytest.raises(Exception):
        model.fit(group_1_counts, group_2_counts, sub_group_labels[2:], verbose=True)
    with pytest.raises(Exception):
        model.fit(group_1_counts, group_2_counts[2:], sub_group_labels, verbose=True)