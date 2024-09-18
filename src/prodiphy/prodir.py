import pymc as pm
import numpy as np
import arviz as az


class ProDir:
    """
    A class to compare two groups using a Bayesian Dirichlet distribution model.

    :param chains: Number of chains to sample in parallel, default is 4
    :param cores: Number of cores to use for sampling, default is 4
    """

    def __init__(self, chains=4, cores=4):
        """
        Initializes the ProDir class with the number of chains and cores.

        :param chains: Number of chains to sample in parallel, default is 4
        :param cores: Number of cores to use for sampling, default is 4
        """
        self.model = None
        self.trace = None
        self.chains = chains
        self.cores = cores

        self.labels = []

    def fit(
        self,
        group_1_data: list[int],
        group_2_data: list[int],
        labels: list[str],
        verbose=False,
    ):
        """
        Fits a Dirichlet distribution model to the two groups and computes the
        difference (delta) and log2 ratio for each category.

        :param group_1_data: List of integer counts for the first group
        :param group_2_data: List of integer counts for the second group
        :param labels: List of category labels corresponding to the data
        :param verbose: If True, prints the input data, default is False

        :raises AssertionError: If the lengths of `group_1_data`, `group_2_data`, and `labels` are not equal
        """
        assert len(group_1_data) == len(group_2_data)
        assert len(group_1_data) == len(labels)

        self.labels = labels

        if verbose:
            print(f"Group 1: {', '.join([str(i) for i in group_1_data])}")
            print(f"Group 2: {', '.join([str(i) for i in group_2_data])}")

        with pm.Model() as self.model:
            group_1_p = pm.Dirichlet("group_1_p", a=np.array(group_1_data) + 1)
            group_2_p = pm.Dirichlet("group_2_p", a=np.array(group_2_data) + 1)

            # Add deterministic variable with difference (delta)
            for ix, label in enumerate(self.labels):
                _ = pm.Deterministic(f"delta_{label}", group_1_p[ix] - group_2_p[ix])
                _ = pm.Deterministic(f"group_1_p_{label}", group_1_p[ix])
                _ = pm.Deterministic(f"group_2_p_{label}", group_2_p[ix])

            # Add deterministic variable with log2 ratio
            for ix, label in enumerate(self.labels):
                _ = pm.Deterministic(
                    f"log2_ratio_{label}", np.log2(group_2_p[ix] / group_1_p[ix])
                )

            self.trace = pm.sample(chains=self.chains, cores=self.cores)

    def get_stats(self):

        var_names = []

        for l in self.labels:
            var_names.append(f"delta_{l}")
            var_names.append(f"group_1_p_{l}")
            var_names.append(f"group_2_p_{l}")
            var_names.append(f"log2_ratio_{l}")

        summary_df = az.summary(self.trace, var_names=var_names)

        return summary_df
