import pymc as pm
import bambi as bmb
import numpy as np
import pandas as pd
import arviz as az


class CorProDir:

    def __init__(self, chains=4, cores=4, draws=1000):
        """
        Initializes the ProDir class with the number of chains and cores.

        :param chains: Number of chains to sample in parallel, default is 4
        :param cores: Number of cores to use for sampling, default is 4
        :param draws:
        """
        self.chains = chains
        self.cores = cores
        self.draws = draws

        self.formula = ""
        self.labels = []
        self.model = None
        self.trace = None
        self.uncorrected_model = None
        self.uncorrected_trace = None
        self.final_data = None

    @staticmethod
    def _create_one_hot_encoding(df, category, labels):
        """
        Converts the categorical column to one-hot encoded columns.

        :param df: DataFrame containing the category.
        :param category: Column name of the categorical variable.
        :param labels: List of unique labels.
        """
        for label in labels:
            df[label] = df[category].apply(lambda x: 1 if x == label else 0)
        return df

    def _get_final_data(self):
        """
        Extracts relevant values from the uncorrected model and glm traces and computes derived values for each label.

        This method retrieves the posterior prevalence from the uncorrected trace and the predicted prevalence
        from the glm trace. It then calculates the delta (difference) and log2 ratio between these values
        for each label. The computed values are stored in the `final_data` DataFrame.
        """
        self.final_data = pd.DataFrame()

        for ix, label in enumerate(self.labels):
            self.final_data[f"{label}_prevalence_dirch"] = (
                self.uncorrected_trace.posterior[f"{label}_prevalence"][0]
            )
            self.final_data[f"{label}_prevalence_est"] = (
                self.trace["posterior_predictive"][f"c({', '.join(self.labels)})"]
                .values[0]
                .mean(axis=1)[:, ix]
            )

            self.final_data[f"{label}_delta"] = (
                self.final_data[f"{label}_prevalence_dirch"]
                - self.final_data[f"{label}_prevalence_est"]
            )
            self.final_data[f"{label}_log2_ratio"] = np.log2(
                self.final_data[f"{label}_prevalence_dirch"]
                / self.final_data[f"{label}_prevalence_est"]
            )

    def _compute_stats(self):
        """
        Computes statistics like mean, standard deviation, HDI, etc. for delta and log2 ratios.

        :return: DataFrame of computed statistics.
        """
        stats = []
        for label in self.labels:
            low_delta, high_delta = az.hdi(np.array(self.final_data[f"{label}_delta"]))
            low_log2_ratio, high_log2_ratio = az.hdi(
                np.array(self.final_data[f"{label}_log2_ratio"])
            )

            stats.append(
                {
                    "label": label,
                    "mean_fraction": np.mean(
                        self.final_data[f"{label}_prevalence_dirch"]
                    ),
                    "mean_estimate": np.mean(
                        self.final_data[f"{label}_prevalence_est"]
                    ),
                    "mean_delta": np.mean(self.final_data[f"{label}_delta"]),
                    "std_delta": np.std(self.final_data[f"{label}_delta"]),
                    "hdi_low_delta": low_delta,
                    "hdi_high_delta": high_delta,
                    "mean_log2_ratio": np.mean(self.final_data[f"{label}_log2_ratio"]),
                    "std_log2_ratio": np.std(self.final_data[f"{label}_log2_ratio"]),
                    "hdi_low_log2_ratio": low_log2_ratio,
                    "hdi_high_log2_ratio": high_log2_ratio,
                    "fraction_above_zero": np.mean(
                        self.final_data[f"{label}_delta"] > 0
                    ),
                    "fraction_below_zero": np.mean(
                        self.final_data[f"{label}_delta"] < 0
                    ),
                }
            )
        return pd.DataFrame(stats)

    def fit(self, reference_df, target_df, category, confounders):
        self.labels = list(
            set(reference_df[category].tolist() + target_df[category].tolist())
        )

        reference_df = self._create_one_hot_encoding(
            reference_df, category, self.labels
        )
        target_df = self._create_one_hot_encoding(target_df, category, self.labels)

        target_counts = target_df.groupby(category).size()[self.labels]

        with pm.Model() as self.uncorrected_model:
            proportions = pm.Dirichlet("proportions", a=target_counts + 1)

            for ix, label in enumerate(self.labels):
                _ = pm.Deterministic(f"{label}_prevalence", proportions[ix])

            self.uncorrected_trace = pm.sample(
                draws=self.draws, chains=self.chains, cores=self.cores
            )

        self.formula = f"c({', '.join(self.labels)}) ~ {' + '.join(confounders)}"

        self.model = bmb.Model(
            self.formula,
            reference_df,
            family="multinomial",
        )
        self.trace = self.model.fit(
            random_seed=0, chains=self.chains, draws=self.draws, cores=self.cores
        )

        # This check was needed, but seems to be unnecessary with the updates in bambi
        # if len(target_df) > 1000:
        #     target_sample = target_df.sample(1000, replace=False)
        # else:
        target_sample = target_df

        self.model.predict(self.trace, data=target_sample, kind="response")

        self._get_final_data()

    def get_stats(self):
        return self._compute_stats()
