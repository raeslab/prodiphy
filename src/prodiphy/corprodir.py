import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm


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
        self.kind = "response_params"
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
            df[label] = df[category].apply(
                lambda x, label=label: 1 if x == label else 0
            )
        return df

    def _get_final_data(self):
        """
        Extracts relevant values from the uncorrected model and glm traces and computes derived values for each label.

        This method retrieves the posterior prevalence from the uncorrected trace and the predicted prevalence
        from the glm trace. It then calculates the delta (difference), log2 ratio and natural log one-vs-rest
        odds ratio between these values for each label. The computed values are stored in the `final_data`
        DataFrame.

        With `kind="response_params"` the predicted prevalence is the average of the predicted class
        probabilities over the target individuals, which is strictly between 0 and 1. With `kind="response"`
        it is the fraction of simulated outcomes, which can be exactly 0 (or 1) for a draw, resulting in
        infinite ratios.
        """
        self.final_data = pd.DataFrame()

        if self.kind == "response_params":
            predicted = self.trace.posterior["p"].values[0].mean(axis=1)
        else:
            predicted = (
                self.trace["posterior_predictive"][f"c({', '.join(self.labels)})"]
                .values[0]
                .mean(axis=1)
            )

        for ix, label in enumerate(self.labels):
            self.final_data[f"{label}_prevalence_dirch"] = (
                self.uncorrected_trace.posterior[f"{label}_prevalence"][0]
            )
            self.final_data[f"{label}_prevalence_est"] = predicted[:, ix]

            self.final_data[f"{label}_delta"] = (
                self.final_data[f"{label}_prevalence_dirch"]
                - self.final_data[f"{label}_prevalence_est"]
            )
            self.final_data[f"{label}_log2_ratio"] = np.log2(
                self.final_data[f"{label}_prevalence_dirch"]
                / self.final_data[f"{label}_prevalence_est"]
            )
            self.final_data[f"{label}_log_odds_ratio"] = self._log_odds(
                self.final_data[f"{label}_prevalence_dirch"]
            ) - self._log_odds(self.final_data[f"{label}_prevalence_est"])

    @staticmethod
    def _log_odds(p):
        """
        Computes the natural log odds, log(p / (1 - p)).

        :param p: Array of probabilities.
        :return: Array of log odds.
        """
        p = np.asarray(p, dtype=float)
        return np.log(p) - np.log1p(-p)

    def _compute_stats(self):
        """
        Computes statistics like mean, standard deviation, HDI, etc. for delta, log2 ratios and odds ratios.

        The one-vs-rest odds ratio compares the odds of a label (vs all other labels) in the target to the
        odds expected from the reference. It is summarized on the natural log scale and back-transformed,
        the odds ratio is reported as the exponentiated posterior median with the exponentiated HDI bounds.

        :return: DataFrame of computed statistics.
        """
        stats = []
        for label in self.labels:
            low_delta, high_delta = az.hdi(np.array(self.final_data[f"{label}_delta"]))
            low_log2_ratio, high_log2_ratio = az.hdi(
                np.array(self.final_data[f"{label}_log2_ratio"])
            )
            log_odds_ratio = np.array(self.final_data[f"{label}_log_odds_ratio"])
            low_log_odds_ratio, high_log_odds_ratio = az.hdi(log_odds_ratio)

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
                    "mean_log_odds_ratio": np.mean(log_odds_ratio),
                    "std_log_odds_ratio": np.std(log_odds_ratio),
                    "hdi_low_log_odds_ratio": low_log_odds_ratio,
                    "hdi_high_log_odds_ratio": high_log_odds_ratio,
                    "odds_ratio": np.exp(np.median(log_odds_ratio)),
                    "hdi_low_odds_ratio": np.exp(low_log_odds_ratio),
                    "hdi_high_odds_ratio": np.exp(high_log_odds_ratio),
                    "fraction_above_zero": np.mean(
                        self.final_data[f"{label}_delta"] > 0
                    ),
                    "fraction_below_zero": np.mean(
                        self.final_data[f"{label}_delta"] < 0
                    ),
                }
            )
        return pd.DataFrame(stats)

    def fit(
        self,
        reference_df,
        target_df,
        category,
        confounders,
        sample_kwargs: dict = None,
        kind: str = "response_params",
    ):
        """
        Fits the uncorrected Dirichlet model on the target and the confounder-corrected multinomial model on the
        reference, and predicts the expected prevalence in the target.

        :param reference_df: DataFrame with the reference cohort.
        :param target_df: DataFrame with the target cohort.
        :param category: Column name of the categorical variable.
        :param confounders: List of column names of the confounders.
        :param sample_kwargs: Additional keyword arguments to pass to the bambi fit.
        :param kind: How the expected prevalence in the target is predicted. "response_params" (default) averages
            the predicted class probabilities, "response" averages simulated outcomes from the posterior predictive.
        """
        if kind not in ("response_params", "response"):
            raise ValueError("kind must be one of 'response_params' or 'response'")
        self.kind = kind

        reference_df = reference_df.copy()
        target_df = target_df.copy()

        self.labels = sorted(
            set(reference_df[category].tolist() + target_df[category].tolist())
        )

        reference_df = self._create_one_hot_encoding(
            reference_df, category, self.labels
        )
        target_df = self._create_one_hot_encoding(target_df, category, self.labels)

        target_counts = (
            target_df.groupby(category).size().reindex(self.labels, fill_value=0)
        )

        if sample_kwargs is None:
            sample_kwargs = {}

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
            random_seed=0,
            chains=self.chains,
            draws=self.draws,
            cores=self.cores,
            **sample_kwargs,
        )

        # This check was needed, but seems to be unnecessary with the updates in bambi
        # if len(target_df) > 1000:
        #     target_sample = target_df.sample(1000, replace=False)
        # else:
        target_sample = target_df

        self.model.predict(self.trace, data=target_sample, kind=self.kind)

        self._get_final_data()

    def get_stats(self):
        return self._compute_stats()
