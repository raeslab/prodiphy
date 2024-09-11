import pandas as pd
import numpy as np

from prodiphy import CorProDir

np.random.seed(1910)


def build_data():
    labels = ["a", "b", "c", "d"]
    ref_size = 200
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
            "age": np.random.randint(18, high=80, size=target_size) + 3,
            "BMI": np.random.normal(25, size=target_size) + 1,
            "label": np.random.choice(
                labels, size=target_size, replace=True, p=target_prevalence
            ),
        }
    )

    return ref_df, target_df


if __name__ == "__main__":
    ref_df, target_df = build_data()

    model = CorProDir(draws=1000)
    output = model.fit(ref_df, target_df, "label", ["age", "BMI"])

    print(output)