import pandas as pd
import pymc as pm
import arviz as az


def _combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame):
    df1["group"] = 0
    df2["group"] = 1

    return pd.concat([df1, df2])


class DeltaSlope:
    def __init__(self, draws=2000, tune=2000, chains=4, cores=4):
        self.model = None
        self.trace = None
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.cores = cores

    def fit(
        self,
        reference_df: pd.DataFrame,
        target_df: pd.DataFrame,
        x: str,
        y: str,
        sample_kwargs: dict = None,
    ):
        df = _combine_dataframes(reference_df[[x, y]], target_df[[x, y]]).dropna()

        if sample_kwargs is None:
            sample_kwargs = {}

        with pm.Model() as self.model:
            x = pm.Data("x", df[x])
            y = pm.Data("y", df[y])
            group = pm.Data("group", df["group"])

            sigmas = pm.Normal("sigmas", 1, shape=2)

            ref_slope = pm.Normal("ref_slope", 0, 2)
            delta_slope = pm.Normal("delta_slope", 0, 1)

            intercepts = pm.Normal("intercepts", 0, 2, shape=2)

            delta_intercept = pm.Deterministic(
                "delta_intercept", intercepts[1] - intercepts[0]
            )
            delta_sigma = pm.Deterministic("delta_sigma", sigmas[1] - sigmas[0])

            ref_intercept = pm.Deterministic("ref_intercept", intercepts[0])
            ref_sigma = pm.Deterministic("ref_sigma", sigmas[0])

            target_slope = pm.Deterministic("target_slope", ref_slope + delta_slope)
            target_intercept = pm.Deterministic("target_intercept", intercepts[1])
            target_sigma = pm.Deterministic("target_sigma", sigmas[1])

            y_obs = pm.Normal(
                "y_obs",
                (ref_slope + (delta_slope * group)) * x + intercepts[group],
                sigmas[group],
                observed=y,
            )

            self.trace = pm.sample(
                self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                **sample_kwargs
            )

    def get_stats(self):
        selected_values = [
            "delta_intercept",
            "delta_sigma",
            "delta_slope",
            "ref_intercept",
            "ref_sigma",
            "ref_slope",
            "target_intercept",
            "target_sigma",
            "target_slope",
        ]

        summary_df = az.summary(self.trace, var_names=selected_values)

        return summary_df
