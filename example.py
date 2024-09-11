import pandas as pd
import numpy as np

from prodiphy import CorProDir

np.random.seed(1910)

labels = ["a", "b", "c", "d"]
def build_data():

    ref_size = 800
    target_size = 100
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

    for label in labels:
        ref_df[label] = ref_df["label"].apply(lambda x: 1 if x == label else 0)
        target_df[label] = target_df["label"].apply(lambda x: 1 if x == label else 0)

    return ref_df, target_df


if __name__ == "__main__":
    ref_df, target_df = build_data()

    model = CorProDir(draws=500)
    output = model.fit(ref_df, target_df, "label", ["age", "BMI"])

    output.to_excel("./tmp/example.xlsx")

    print(model.formula)