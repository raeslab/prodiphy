import re
from typing import Literal

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

# A covariate specification is a (column_name, encoding) tuple where encoding is
# either the string "continuous" (z-scored before fitting) or a dict mapping the
# observed values to a 0/1 contrast (an explicit, documented binary encoding).
CovariateSpec = tuple[str, str | dict]


class ACORN:
    """Adjusted CORrelations, Negative-binomial.

    A hierarchical Negative-Binomial model that tests, for *every* feature
    (taxon) in a count table simultaneously, whether its abundance is associated
    with a continuous marker, while adjusting for an arbitrary set of covariates.

    All features are fit jointly in a single model. Each feature gets its own
    intercept and slopes, but those per-feature coefficients are partially pooled
    toward a shared group distribution via a non-centered hierarchical prior. This
    shrinks noisy/rare features toward the population mean (a principled
    alternative to per-feature GLMs + FDR correction) while letting well-measured
    features keep their own signal, and yields a full posterior per feature.

    For feature ``t`` of row ``i`` (one row per (sample, feature) pair)::

        y_i ~ NegativeBinomial(mu = exp(log_mu_i), alpha = dispersion[t])
        log_mu_i = intercept[t] + beta_marker[t] * marker_i
                                + sum_c beta_c[t] * covariate_c_i
                                + offset_i

    Every per-feature coefficient is non-centered: ``X[t] = mu_X + tau_X * z_X[t]``
    with ``mu_X ~ Normal(0, 1)``, ``tau_X ~ HalfNormal(1)``, ``z_X[t] ~ Normal(0, 1)``.

    :param chains: Number of chains to sample in parallel, default is 3
    :param cores: Number of cores to use for sampling, default is 4
    :param draws: Number of posterior draws (NUTS), default is 500
    :param tune: Number of tuning steps (NUTS), default is 500
    :param target_accept: Target acceptance rate for NUTS, default is 0.9
    :param random_seed: Random seed for reproducible fits, default is 42
    :param advi_iterations: Number of optimization steps for ADVI, default is 30000
    :param advi_draws: Number of draws taken from the fitted ADVI approximation, default is 1000
    """

    def __init__(
        self,
        chains: int = 3,
        cores: int = 4,
        draws: int = 500,
        tune: int = 500,
        target_accept: float = 0.9,
        random_seed: int = 42,
        advi_iterations: int = 30000,
        advi_draws: int = 1000,
    ):
        self.chains = chains
        self.cores = cores
        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.advi_iterations = advi_iterations
        self.advi_draws = advi_draws

        self.model = None
        self.trace = None
        self.feature_names = []
        self.marker = None
        # Ordered list of (column_name, pymc_var_name) for the covariates.
        self.covariate_vars = []
        # mean/std used to z-score the marker, kept for back-transformation.
        self.marker_mean = None
        self.marker_std = None

    @staticmethod
    def _standardize(values: np.ndarray, name: str):
        """Z-score a 1-D array, returning the scaled values, mean and std."""
        values = np.asarray(values, dtype=float)
        mean, std = np.nanmean(values), np.nanstd(values)
        if std == 0 or not np.isfinite(std):
            raise ValueError(
                f"Cannot standardize '{name}': it has zero or undefined variance."
            )
        return (values - mean) / std, mean, std

    @staticmethod
    def _var_name(prefix: str, column: str) -> str:
        """Build a clean PyMC variable name from a covariate column name."""
        return f"{prefix}{re.sub(r'[^0-9a-zA-Z_]', '_', column)}"

    def _encode_covariate(self, df: pd.DataFrame, spec: CovariateSpec) -> np.ndarray:
        """Encode a single covariate column according to its declared encoding."""
        column, encoding = spec
        series = df[column]

        if encoding == "continuous":
            scaled, _, _ = self._standardize(series.to_numpy(dtype=float), column)
            return scaled

        if isinstance(encoding, dict):
            observed = set(pd.unique(series.dropna()))
            unmapped = observed - set(encoding.keys())
            if unmapped:
                raise ValueError(
                    f"Covariate '{column}' has values {sorted(map(str, unmapped))} "
                    f"that are not in the supplied encoding {encoding}."
                )
            codes = set(encoding.values())
            if not codes <= {0, 1}:
                raise ValueError(
                    f"Encoding for covariate '{column}' must map to a 0/1 binary "
                    f"contrast, got codes {sorted(map(str, codes))}. Multi-level "
                    "(k>2) categorical covariates are not supported; encode them "
                    "as separate binary covariates instead."
                )
            return series.map(encoding).to_numpy(dtype=float)

        raise ValueError(
            f"Encoding for covariate '{column}' must be 'continuous' or a "
            f"{{value: 0/1}} dict, got {encoding!r}."
        )

    def _build_model_data(
        self,
        data: pd.DataFrame,
        count_cols: list[str],
        marker: str,
        covariates: list[CovariateSpec],
        offset: bool,
        offset_col: str,
    ) -> dict:
        """Reshape a wide sample x feature table into long model-ready arrays."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame.")
        if data.empty:
            raise ValueError("Input data is empty.")
        if len(count_cols) == 0:
            raise ValueError("count_cols must contain at least one feature column.")

        covariate_cols = [column for column, _ in covariates]
        required = [marker, *covariate_cols, *count_cols]
        if offset and offset_col is not None:
            required.append(offset_col)
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        # Drop samples missing the marker or any covariate so every feature row
        # for a given sample is built from complete predictor data.
        needed = [marker, *covariate_cols]
        if offset and offset_col is not None:
            needed.append(offset_col)
        df = data.dropna(subset=needed).reset_index(drop=True)
        if df.empty:
            raise ValueError("No samples remain after dropping missing predictors.")

        feature_names = list(count_cols)
        n_features = len(feature_names)
        n_samples = len(df)

        # Wide -> long: each sample contributes one row per feature, with the
        # feature block as the fastest-varying axis (sample-major layout).
        abundances = df[feature_names].to_numpy(dtype=float)
        y = abundances.reshape(-1)
        feature_idx = np.tile(np.arange(n_features), n_samples)

        marker_std, self.marker_mean, self.marker_std = self._standardize(
            df[marker].to_numpy(dtype=float), marker
        )

        def repeat_per_sample(v):
            return np.repeat(v, n_features)

        model_data = {
            "y": y,
            "feature_idx": feature_idx,
            "feature_names": feature_names,
            "marker": repeat_per_sample(marker_std),
        }

        self.covariate_vars = []
        for spec in covariates:
            column = spec[0]
            var_name = self._var_name("beta_", column)
            model_data[var_name] = repeat_per_sample(self._encode_covariate(df, spec))
            self.covariate_vars.append((column, var_name))

        if offset:
            if offset_col is not None:
                totals = df[offset_col].to_numpy(dtype=float)
            else:
                totals = abundances.sum(axis=1)
            if np.any(totals <= 0):
                raise ValueError(
                    "Offset requires strictly positive sample totals; found "
                    "zero or negative totals."
                )
            model_data["offset"] = repeat_per_sample(np.log(totals))

        return model_data

    def fit(
        self,
        data: pd.DataFrame,
        count_cols: list[str],
        marker: str,
        covariates: list[CovariateSpec] = None,
        method: Literal["nuts", "advi", "fullrank_advi"] = "nuts",
        nuts_sampler: str = "pymc",
        offset: bool = False,
        offset_col: str = None,
        hierarchical_dispersion: bool = False,
        sample_kwargs: dict = None,
    ):
        """Fit the hierarchical NB model relating one marker to all features.

        :param data: Wide DataFrame, one row per sample, with the feature count
            columns, the marker column and any covariate columns.
        :param count_cols: List of column names holding the (integer) feature counts.
        :param marker: Column name of the continuous marker of interest (z-scored).
        :param covariates: List of ``(column, encoding)`` specs. ``encoding`` is
            either ``"continuous"`` (z-scored) or a ``{value: 0/1}`` dict giving an
            explicit binary contrast. Multi-level (k>2) categoricals are rejected.
        :param method: Inference engine: ``"nuts"`` for the publication-quality
            posterior, ``"advi"``/``"fullrank_advi"`` for a fast variational preview
            (these underestimate uncertainty in hierarchical models).
        :param nuts_sampler: NUTS backend passed to ``pm.sample``, default ``"pymc"``.
            Pass ``"numpyro"`` (if installed) for GPU-accelerated sampling.
        :param offset: If True, add a ``log(total)`` offset to the linear predictor
            for count tables that are not normalized to even sequencing depth.
        :param offset_col: Column to use for the offset total. If None and
            ``offset`` is True, the per-sample sum of ``count_cols`` is used.
        :param hierarchical_dispersion: If True, partially pool the per-feature
            dispersion on the log scale; if False (default), each feature gets an
            independent ``HalfNormal(1)`` dispersion.
        :param sample_kwargs: Extra keyword arguments forwarded to ``pm.sample``.
        """
        if covariates is None:
            covariates = []
        if sample_kwargs is None:
            sample_kwargs = {}

        model_data = self._build_model_data(
            data, count_cols, marker, covariates, offset, offset_col
        )
        self.marker = marker
        self.feature_names = model_data["feature_names"]
        feature_idx = model_data["feature_idx"]

        coords = {"taxon": self.feature_names}

        with pm.Model(coords=coords) as self.model:
            # Each predictor gets the same hierarchical treatment via a closure:
            # a group mean, a group spread, and feature-specific non-centered
            # offsets. Non-centering (mu + tau * z) keeps NUTS healthy when a
            # group spread is small, which is the common case here.
            def hierarchical_slope(name):
                group_mean = pm.Normal(f"mu_{name}", 0.0, 1.0)
                group_spread = pm.HalfNormal(f"tau_{name}", 1.0)
                offsets = pm.Normal(f"z_{name}", 0.0, 1.0, dims="taxon")
                return pm.Deterministic(
                    name, group_mean + group_spread * offsets, dims="taxon"
                )

            intercept = hierarchical_slope("intercept")
            beta_marker = hierarchical_slope("beta_marker")

            log_mu = (
                intercept[feature_idx] + beta_marker[feature_idx] * model_data["marker"]
            )

            for _, var_name in self.covariate_vars:
                beta_cov = hierarchical_slope(var_name)
                log_mu = log_mu + beta_cov[feature_idx] * model_data[var_name]

            if "offset" in model_data:
                log_mu = log_mu + model_data["offset"]

            # Per-feature overdispersion: rare features are noisier than abundant
            # ones. Optionally partially pooled on the log scale.
            if hierarchical_dispersion:
                mu_disp = pm.Normal("mu_dispersion", 0.0, 1.0)
                tau_disp = pm.HalfNormal("tau_dispersion", 1.0)
                z_disp = pm.Normal("z_dispersion", 0.0, 1.0, dims="taxon")
                dispersion = pm.Deterministic(
                    "dispersion", pm.math.exp(mu_disp + tau_disp * z_disp), dims="taxon"
                )
            else:
                dispersion = pm.HalfNormal("dispersion", 1.0, dims="taxon")

            pm.NegativeBinomial(
                "y_obs",
                mu=pm.math.exp(log_mu),
                alpha=dispersion[feature_idx],
                observed=model_data["y"],
            )

            if method == "advi":
                approx = pm.fit(
                    n=self.advi_iterations, method="advi", random_seed=self.random_seed
                )
                self.trace = approx.sample(self.advi_draws)
            elif method == "fullrank_advi":
                approx = pm.fit(
                    n=self.advi_iterations,
                    method="fullrank_advi",
                    random_seed=self.random_seed,
                )
                self.trace = approx.sample(self.advi_draws)
            elif method == "nuts":
                self.trace = pm.sample(
                    draws=self.draws,
                    tune=self.tune,
                    chains=self.chains,
                    cores=self.cores,
                    target_accept=self.target_accept,
                    random_seed=self.random_seed,
                    nuts_sampler=nuts_sampler,
                    **sample_kwargs,
                )
            else:
                raise ValueError(
                    f"method must be 'nuts', 'advi', or 'fullrank_advi', got {method!r}"
                )

        return self.trace

    def get_stats(self, rope_log: float = 0.05, hdi_prob: float = 0.94) -> pd.DataFrame:
        """Turn the per-feature marker-slope posterior into a ranked table.

        This is the scientific output of the model: one row per feature, ranked by
        strength of evidence, with the adjusted log-effect of the marker, its HDI,
        a fold-change and two posterior-probability summaries.

        :param rope_log: Half-width of the "region of practical equivalence" on the
            log scale. ``|effect| < rope_log`` counts as no meaningful association
            (default 0.05 ≈ a 5% abundance change per 1 SD of the marker).
        :param hdi_prob: Probability mass of the highest-density interval, default 0.94.
        :return: A ranked DataFrame with columns ``feature``, ``log_effect``,
            ``hdi_<p>_low/high``, ``fold_change``, ``prob_direction``,
            ``prob_outside_rope`` and ``hdi_excludes_zero``.
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been fitted yet.")

        beta = self.trace.posterior["beta_marker"]
        samples = (
            beta.stack(sample=("chain", "draw")).transpose("taxon", "sample").to_numpy()
        )

        median = np.median(samples, axis=1)
        hdi = az.hdi(self.trace, var_names=["beta_marker"], hdi_prob=hdi_prob)[
            "beta_marker"
        ].to_numpy()

        # Probability of direction: how lopsided the posterior is around zero
        # (0.5 = no evidence of direction, 1.0 = certain). The Bayesian analog of
        # "is this non-zero", replacing the FDR p-value.
        prob_positive = np.mean(samples > 0, axis=1)
        prob_direction = np.maximum(prob_positive, 1 - prob_positive)

        # Fraction of posterior mass outside the practically-null band.
        prob_outside_rope = np.mean(np.abs(samples) > rope_log, axis=1)

        hdi_pct = int(hdi_prob * 100)
        table = pd.DataFrame(
            {
                "feature": self.feature_names,
                "log_effect": median,
                f"hdi_{hdi_pct}_low": hdi[:, 0],
                f"hdi_{hdi_pct}_high": hdi[:, 1],
                "fold_change": np.exp(median),
                "prob_direction": prob_direction,
                "prob_outside_rope": prob_outside_rope,
            }
        )

        hdi_low = table[f"hdi_{hdi_pct}_low"]
        hdi_high = table[f"hdi_{hdi_pct}_high"]
        table["hdi_excludes_zero"] = (hdi_low > 0) | (hdi_high < 0)

        table = table.sort_values(
            ["prob_direction", "log_effect"],
            key=lambda col: col.abs() if col.name == "log_effect" else col,
            ascending=False,
        ).reset_index(drop=True)

        return table

    def get_summary(self, var_names: list[str] = None) -> pd.DataFrame:
        """Return a raw ArviZ posterior summary (diagnostics, means, HDIs).

        Useful for sampler diagnostics and for inspecting the group-level
        ``mu_*``/``tau_*`` parameters. Defaults to summarizing ``beta_marker``.

        :param var_names: Variables to summarize, default ``["beta_marker"]``.
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been fitted yet.")
        if var_names is None:
            var_names = ["beta_marker"]
        return az.summary(self.trace, var_names=var_names)
