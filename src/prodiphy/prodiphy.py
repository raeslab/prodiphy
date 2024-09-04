import pymc as pm
import numpy as np


class ProDir:
    def __init__(self, chains=4, cores=4):
        self.model = None
        self.trace = None
        self.chains = chains
        self.cores = cores

    def fit(self, group_1_data: list[int], group_2_data: list[int], labels):
        print(f"Group 1: {', '.join([str(i) for i in group_1_data])}")
        print(f"Group 2: {', '.join([str(i) for i in group_2_data])}")

        with pm.Model() as self.model:
            group_1_p = pm.Dirichlet("group_1_p", a=np.array(group_1_data) + 1)
            group_2_p = pm.Dirichlet("group_2_p", a=np.array(group_2_data) + 1)

            # Add deterministic variable with difference (delta)
            for ix, label in enumerate(labels):
                _ = pm.Deterministic(f"delta_{label}", group_1_p[ix] - group_2_p[ix])

            # Add deterministic variable with log2 ratio
            for ix, label in enumerate(labels):
                _ = pm.Deterministic(
                    f"log2_ratio_{label}", np.log2(group_2_p[ix] / group_1_p[ix])
                )

            self.trace = pm.sample(chains=self.chains, cores=self.cores)
