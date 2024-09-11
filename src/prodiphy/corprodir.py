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
        """
        self.chains = chains
        self.cores = cores
        self.draws = draws

        self.formula = ""
        self.model = None
        self.trace = None
        self.uncorrected_model = None
        self.uncorrected_trace = None

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

    @staticmethod
    def _compute_stats(final_data, labels):
        """
        Computes statistics like mean, standard deviation, HDI, etc. for delta and log2 ratios.

        :param final_data: DataFrame containing prevalence data.
        :param labels: List of unique labels.
        :return: DataFrame of computed statistics.
        """
        stats = []
        for label in labels:
            low_delta, high_delta = az.hdi(np.array(final_data[f"{label}_delta"]))
            low_log2_ratio, high_log2_ratio = az.hdi(np.array(final_data[f"{label}_log2_ratio"]))

            stats.append(
                {
                    "label": label,
                    "mean_fraction": np.mean(final_data[f"{label}_prevalence_dirch"]),
                    "mean_estimate": np.mean(final_data[f"{label}_prevalence_est"]),
                    "mean_delta": np.mean(final_data[f"{label}_delta"]),
                    "std_delta": np.std(final_data[f"{label}_delta"]),
                    "hdi_low_delta": low_delta,
                    "hdi_high_delta": high_delta,
                    "mean_log2_ratio": np.mean(final_data[f"{label}_log2_ratio"]),
                    "std_log2_ratio": np.std(final_data[f"{label}_log2_ratio"]),
                    "hdi_low_log2_ratio": low_log2_ratio,
                    "hdi_high_log2_ratio": high_log2_ratio,
                    "fraction_above_zero": np.mean(final_data[f"{label}_delta"] > 0),
                    "fraction_below_zero": np.mean(final_data[f"{label}_delta"] < 0),
                }
            )
        return pd.DataFrame(stats)

    def fit(self, reference_df, target_df, category, confounders):
        labels = list(
            set(reference_df[category].tolist() + target_df[category].tolist())
        )

        reference_df = self._create_one_hot_encoding(reference_df, category, labels)
        target_df = self._create_one_hot_encoding(target_df, category, labels)

        target_counts = target_df.groupby(category).size()[labels]

        with pm.Model() as self.uncorrected_model:
            proportions = pm.Dirichlet("proportions", a=target_counts + 1)

            for ix, label in enumerate(labels):
                _ = pm.Deterministic(f"{label}_prevalence", proportions[ix])

            self.uncorrected_trace = pm.sample(
                draws=self.draws, chains=self.chains, cores=self.cores
            )

        self.formula = f"c({', '.join(labels)}) ~ {' + '.join(confounders)}"

        self.model = bmb.Model(
            self.formula,
            reference_df,
            family="multinomial",
        )
        self.trace = self.model.fit(
            random_seed=0, chains=self.chains, draws=self.draws, cores=self.cores
        )

        sample_size = min(len(reference_df), 1000)
        target_sample = target_df.sample(sample_size, replace=len(reference_df) < 1000)
        self.model.predict(self.trace, data=target_sample, kind="response")

        final_data = pd.DataFrame()

        for ix, label in enumerate(labels):
            final_data[f"{label}_prevalence_dirch"] = self.uncorrected_trace.posterior[
                f"{label}_prevalence"
            ][0]
            final_data[f"{label}_prevalence_est"] = (
                self.trace["posterior_predictive"][f"c({', '.join(labels)})"]
                .values[0]
                .mean(axis=1)[:, ix]
            )

            final_data[f"{label}_delta"] = (
                final_data[f"{label}_prevalence_dirch"]
                - final_data[f"{label}_prevalence_est"]
            )
            final_data[f"{label}_log2_ratio"] = np.log2(
                final_data[f"{label}_prevalence_dirch"]
                / final_data[f"{label}_prevalence_est"]
            )

        return self._compute_stats(final_data, labels)
