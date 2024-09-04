import arviz as az
import pytest
from prodiphy import ProDir


def test_prodir():
    model = ProDir()

    group_1_counts = [200, 300, 150, 100]
    group_2_counts = [10, 33, 12, 8]
    sub_group_labels = ["a", "b", "c", "d"]

    model.fit(group_1_counts, group_2_counts, sub_group_labels)

    with model.model:
        summary_df = az.summary(model.trace)

    for l in sub_group_labels:
        assert f"delta_{l}" in summary_df.index
        assert f"log2_ratio_{l}" in summary_df.index

    for ix, v in enumerate(group_1_counts):
        proportion_obs = v/sum(group_1_counts)
        assert f"group_1_p[{ix}]" in summary_df.index
        assert summary_df.loc[f"group_1_p[{ix}]", "mean"] == pytest.approx(proportion_obs, rel=1e-1)

    for ix, v in enumerate(group_2_counts):
        proportion_obs = v/sum(group_2_counts)
        assert f"group_2_p[{ix}]" in summary_df.index
        assert summary_df.loc[f"group_2_p[{ix}]", "mean"] == pytest.approx(proportion_obs, rel=1e-1)
