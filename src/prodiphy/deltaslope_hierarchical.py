import arviz as az
import pandas as pd
import pymc as pm


def _combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df1.assign(group=0), df2.assign(group=1)], ignore_index=True)


class DeltaSlopeHierarchical:
    """Compare linear regressions between two groups with hierarchical shrinkage.

    The slopes and intercepts of the reference and target group are drawn from a
    shared (non-centered) population distribution, so differences between groups
    are shrunk towards zero unless the data supports them. The model is fitted on
    standardized x and y (z-scores of the pooled data); all reported parameters are
    transformed back to the original scale.

    Note that with only two groups, the between-group standard deviations
    (``sigma_slope`` and ``sigma_intercept``) are informed mostly by their priors,
    so ``sigma_slope_prior`` and ``sigma_intercept_prior`` set how strongly the
    groups are pulled together. They are expressed on the standardized scale.
    """

    def __init__(
        self,
        draws=2000,
        tune=2000,
        chains=4,
        cores=4,
        sigma_slope_prior=0.5,
        sigma_intercept_prior=0.5,
    ):
        self.model = None
        self.trace = None
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.cores = cores
        self.sigma_slope_prior = sigma_slope_prior
        self.sigma_intercept_prior = sigma_intercept_prior

    def fit(
        self,
        reference_df: pd.DataFrame,
        target_df: pd.DataFrame,
        x: str,
        y: str,
        sample_kwargs: dict = None,
    ):
        df = _combine_dataframes(reference_df[[x, y]], target_df[[x, y]]).dropna()

        x_mean, x_std = df[x].mean(), df[x].std()
        y_mean, y_std = df[y].mean(), df[y].std()

        if not (x_std > 0 and y_std > 0):
            raise ValueError(
                f"Columns '{x}' and '{y}' must both vary to fit a regression."
            )

        x_scaled = ((df[x] - x_mean) / x_std).to_numpy()
        y_scaled = ((df[y] - y_mean) / y_std).to_numpy()

        # Hierarchical models with two groups are prone to divergences, sample more carefully
        sample_kwargs = {"target_accept": 0.95, **(sample_kwargs or {})}

        coords = {"group": ["reference", "target"]}

        with pm.Model(coords=coords) as self.model:
            x_data = pm.Data("x", x_scaled)
            y_data = pm.Data("y", y_scaled)
            group = pm.Data("group", df["group"].to_numpy())

            # Non-centered hierarchical slopes (standardized scale)
            mu_slope = pm.Normal("mu_slope", 0, 1)
            sigma_slope = pm.HalfNormal("sigma_slope", self.sigma_slope_prior)
            z_slope = pm.Normal("z_slope", 0, 1, dims="group")
            slopes = pm.Deterministic(
                "slopes_std", mu_slope + z_slope * sigma_slope, dims="group"
            )

            # Non-centered hierarchical intercepts; as x is centered, these are the
            # expected values of y at the pooled mean of x (standardized scale)
            mu_intercept = pm.Normal("mu_intercept", 0, 1)
            sigma_intercept = pm.HalfNormal(
                "sigma_intercept", self.sigma_intercept_prior
            )
            z_intercept = pm.Normal("z_intercept", 0, 1, dims="group")
            intercepts = pm.Deterministic(
                "intercepts_std",
                mu_intercept + z_intercept * sigma_intercept,
                dims="group",
            )

            sigmas = pm.HalfNormal("sigmas_std", 1, dims="group")

            pm.Normal(
                "y_obs",
                slopes[group] * x_data + intercepts[group],
                sigmas[group],
                observed=y_data,
            )

            # Back-transform to the original scale
            slope_orig = slopes * y_std / x_std
            at_mean_x = y_mean + intercepts * y_std
            intercept_orig = at_mean_x - slope_orig * x_mean
            sigma_orig = sigmas * y_std

            for i, prefix in enumerate(["ref", "target"]):
                pm.Deterministic(f"{prefix}_slope", slope_orig[i])
                pm.Deterministic(f"{prefix}_intercept", intercept_orig[i])
                pm.Deterministic(f"{prefix}_sigma", sigma_orig[i])
                pm.Deterministic(f"{prefix}_at_mean_x", at_mean_x[i])

            pm.Deterministic("delta_slope", slope_orig[1] - slope_orig[0])
            pm.Deterministic("delta_intercept", intercept_orig[1] - intercept_orig[0])
            pm.Deterministic("delta_sigma", sigma_orig[1] - sigma_orig[0])
            pm.Deterministic("delta_at_mean_x", at_mean_x[1] - at_mean_x[0])

            self.trace = pm.sample(
                self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.cores,
                **sample_kwargs,
            )

    def get_stats(self):
        selected_values = [
            "delta_at_mean_x",
            "delta_intercept",
            "delta_sigma",
            "delta_slope",
            "ref_at_mean_x",
            "ref_intercept",
            "ref_sigma",
            "ref_slope",
            "target_at_mean_x",
            "target_intercept",
            "target_sigma",
            "target_slope",
        ]

        summary_df = az.summary(self.trace, var_names=selected_values)

        return summary_df
