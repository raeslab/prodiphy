import pandas as pd
import numpy as np

from prodiphy import CorProDir

np.random.seed(1910)


def build_data(labels=["a", "b", "c", "d"]):
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

    return ref_df, target_df


def test_corprodir():
    labels = ["a", "b", "c", "d"]
    n_draws = 200
    ref_df, target_df = build_data(labels)

    model = CorProDir(draws=n_draws)

    stats_df = model.fit(ref_df, target_df, "label", ["age", "BMI"])

    assert type(stats_df) == pd.DataFrame
    assert stats_df.shape[0] == len(labels)   # should match the number of label in the dataset
    assert stats_df.shape[1] == 15

    for label in labels:
        assert label in stats_df["label"].tolist()
