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

    def fit(self, reference_df, target_df, category, confounders):
        labels = list(
            set(reference_df[category].tolist() + target_df[category].tolist())
        )

        for label in labels:
            reference_df[label] = reference_df[category].apply(
                lambda x: 1 if x == label else 0
            )
            target_df[label] = target_df[category].apply(
                lambda x: 1 if x == label else 0
            )

        # Get uncorrected ET proportions
        target_counts = target_df.groupby(category).size()[labels]

        with pm.Model() as self.uncorrected_model:
            proportions = pm.Dirichlet("proportions", a=target_counts + 1)

            for ix, label in enumerate(labels):
                _ = pm.Deterministic(f"{label}_prevalence", proportions[ix])

            self.uncorrected_trace = pm.sample(
                draws=self.draws, chains=self.chains, cores=self.cores
            )

        # make ET prevalence model on good stratum
        self.formula = f"c({', '.join(labels)}) ~ {' + '.join(confounders)}"

        self.model = bmb.Model(
            self.formula,
            reference_df,
            family="multinomial",
        )
        self.trace = self.model.fit(
            random_seed=0, chains=self.chains, draws=self.draws, cores=self.cores
        )

        if len(target_df) < 1000:
            self.model.predict(
                self.trace,
                data=target_df.sample(reference_df.shape[0], replace=True),
                kind="response",
            ) # the number samples needs to match the reference for this prediction to work
        else:
            self.model.predict(
                self.trace, data=target_df.sample(n=1000, replace=False), kind="response"
            )

        mfinal_data = pd.DataFrame()

        for ix, label in enumerate(labels):
            mfinal_data[f"{label}_prevalence_dirch"] = self.uncorrected_trace.posterior[
                f"{label}_prevalence"
            ][0]
            mfinal_data[f"{label}_prevalence_est"] = (
                self.trace["posterior_predictive"][f"c({', '.join(labels)})"]
                .values[0]
                .mean(axis=1)[:, ix]
            )

            mfinal_data[f"{label}_delta"] = (
                mfinal_data[f"{label}_prevalence_dirch"]
                - mfinal_data[f"{label}_prevalence_est"]
            )
            mfinal_data[f"{label}_log2_ratio"] = np.log2(
                mfinal_data[f"{label}_prevalence_dirch"]
                / mfinal_data[f"{label}_prevalence_est"]
            )

        stats = []

        for label in labels:
            low_delta, high_delta = az.hdi(np.array(mfinal_data[f"{label}_delta"]))
            low_log2_ratio, high_log2_ratio = az.hdi(
                np.array(mfinal_data[f"{label}_log2_ratio"])
            )

            stats.append(
                {
                    "label": label,
                    "n_ref": len(reference_df),
                    "n_target": len(target_df),
                    "mean_fraction": np.mean(mfinal_data[f"{label}_prevalence_dirch"]),
                    "mean_estimate": np.mean(mfinal_data[f"{label}_prevalence_est"]),
                    "mean_delta": np.mean(mfinal_data[f"{label}_delta"]),
                    "std_delta": np.std(mfinal_data[f"{label}_delta"]),
                    "hdi_low_delta": low_delta,
                    "hdi_high_delta": high_delta,
                    "mean_log2_ratio": np.mean(mfinal_data[f"{label}_log2_ratio"]),
                    "std_log2_ratio": np.std(mfinal_data[f"{label}_log2_ratio"]),
                    "hdi_low_log2_ratio": low_log2_ratio,
                    "hdi_high_log2_ratio": high_log2_ratio,
                    "fraction_above_zero": mfinal_data[f"{label}_delta"]
                    .apply(lambda x: 1 if x > 0 else 0)
                    .mean(),
                    "fraction_below_zero": mfinal_data[f"{label}_delta"]
                    .apply(lambda x: 1 if x < 0 else 0)
                    .mean(),
                }
            )

        return pd.DataFrame(stats)
