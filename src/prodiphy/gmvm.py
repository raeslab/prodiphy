import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
import pytensor.tensor as pt

from typing import Literal


class GMvM:
    def __init__(self, clusters: int, chains=4, cores=4, samples=1000, tune=1500):
        """
        Initializes the GMvM (Gaussian Multivariate Mixture) class with the number of clusters
        and sampling parameters.

        :param clusters: Number of clusters to include in the model
        :param chains: Number of chains to sample in parallel, default is 4
        :param cores: Number of cores to use for sampling, default is 4
        :param samples: Number of samples to draw, default is 1000
        :param tune: Number of tuning steps, default is 1500
        """
        if clusters < 1:
            raise ValueError("Number of clusters must be greater than 0.")

        self.clusters = clusters
        self.model = None
        self.trace = None
        self.chains = chains
        self.cores = cores
        self.samples = samples
        self.tune = tune
        self.obs = None  # Store for cluster assignment

    def _prepare_data(self, data):
        """Prepares and validates input data."""
        if isinstance(data, pd.DataFrame):
            if data.empty:
                raise ValueError("Input data is empty.")
            data_array = data.values
        elif isinstance(data, np.ndarray):
            if data.size == 0:
                raise ValueError("Input data is empty.")
            data_array = data
        else:
            raise ValueError("Data must be a pandas DataFrame or numpy array.")

        if len(data_array.shape) != 2:
            raise ValueError("Data must be 2-dimensional (observations x features).")

        n_observations, n_features = data_array.shape

        if n_observations < self.clusters:
            raise ValueError("Number of observations must be greater than number of clusters.")

        return data_array, n_observations, n_features

    def fit(
        self,
        data,
        eta: float = 2.0,
        sd: float = 1.0,
        mu_prior_std: float = 1.5,
        sample_kwargs=None,
    ):
        """
        Fits a Gaussian Multivariate Mixture model to the input data.

        :param data: A pandas DataFrame or numpy array with continuous values
        :param eta: LKJ prior parameter for correlation matrices, default is 2.0
        :param sd: Standard deviation for the half-normal prior on scale parameters, default is 1.0
        :param mu_prior_std: Standard deviation for the normal prior on cluster means, default is 1.5
        :param sample_kwargs: Additional keyword arguments to pass to pm.sample()
        """
        data_array, n_observations, n_features = self._prepare_data(data)

        if sample_kwargs is None:
            sample_kwargs = {}

        with pm.Model() as self.model:
            # Create individual covariance matrices and means for each cluster
            chols = []
            mus = []

            for k in range(self.clusters):
                # Create covariance matrix for cluster k
                chol_k, corr_k, stds_k = pm.LKJCholeskyCov(
                    f"sigma_{k}",
                    n=n_features,
                    eta=eta,
                    sd_dist=pm.HalfNormal.dist(sigma=sd),
                    compute_corr=True,
                )
                chols.append(chol_k)

                # Create mean vector for cluster k
                mu_k = pm.Normal(f"mu_{k}", 0.0, mu_prior_std, shape=n_features)
                mus.append(mu_k)

            # Create the multivariate normal distribution for each cluster
            MultivariateNormals = [
                pm.MvNormal.dist(mus[k], chol=chols[k], shape=n_features)
                for k in range(self.clusters)
            ]

            # Create the weights for each cluster
            w = pm.Dirichlet("w", np.ones(self.clusters) / self.clusters)

            self.obs = pm.Mixture("obs", w=w, comp_dists=MultivariateNormals, observed=data_array)

            self.trace = pm.sample(
                self.samples,
                cores=self.cores,
                tune=self.tune,
                chains=self.chains,
                target_accept=0.90,
                idata_kwargs={"log_likelihood": True},
                **sample_kwargs,
            )

    def _build_recovery_model(self, data_array, n_features):
        """
        Build a model for recovering cluster assignments.
        Similar to the approach used in DMM.
        """
        with pm.Model() as recovery_model:
            # Recreate the model structure for cluster assignment
            chols = []
            mus = []

            for k in range(self.clusters):
                # Create covariance matrix for cluster k
                chol_k, _, _ = pm.LKJCholeskyCov(
                    f"sigma_{k}",
                    n=n_features,
                    eta=2.0,
                    sd_dist=pm.HalfNormal.dist(sigma=1.0),
                    compute_corr=True,
                )
                chols.append(chol_k)

                # Create mean vector for cluster k
                mu_k = pm.Normal(f"mu_{k}", 0.0, 1.5, shape=n_features)
                mus.append(mu_k)

            # Create the multivariate normal distributions
            components = [
                pm.MvNormal.dist(mus[k], chol=chols[k], shape=n_features)
                for k in range(self.clusters)
            ]

            # Create weights
            w = pm.Dirichlet("w", np.ones(self.clusters) / self.clusters)

            # Compute log probabilities for each component
            log_probs = pt.concatenate(
                [
                    [pt.log(w[i]) + pm.logp(components[i], data_array)]
                    for i in range(self.clusters)
                ],
                axis=0,
            )

            # Create categorical variable for cluster assignment
            _ = pm.Categorical("idx", logit_p=log_probs.T)

        return recovery_model

    def get_clusters(self, data, chain_idx=0):
        """
        Assigns clusters to the input data based on the fitted Gaussian Multivariate Mixture model.

        :param data: A pandas DataFrame or numpy array with continuous values
        :param chain_idx: Index of the chain to use for cluster assignment, default is 0
        :return: A pandas DataFrame with cluster assignments and probabilities for each cluster
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been fitted yet.")

        data_array, n_observations, n_features = self._prepare_data(data)

        # Build recovery model
        recovery_model = self._build_recovery_model(data_array, n_features)

        # Sample posterior predictive for cluster assignments
        with recovery_model:
            pp = pm.sample_posterior_predictive(self.trace, var_names=["idx"])
            idx = pp.posterior_predictive["idx"]

        n_draws = idx.shape[1]
        output = []

        for i in range(n_observations):
            cluster_ids = np.array(idx.sel(chain=chain_idx).T[i])
            probs = np.array(
                [sum(cluster_ids == n) / n_draws for n in range(self.clusters)]
            )
            cluster = f"C{probs.argmax() + 1}"
            output.append(
                {
                    **{f"C{n + 1}": prob for n, prob in enumerate(probs)},
                    "idx": i,
                    "cluster": cluster,
                }
            )

        return pd.DataFrame(output).set_index("idx", drop=True)

    def get_stats(self, chain_idx=0):
        """
        Returns a summary of the trace in ArviZ format. As different clusters may have different
        indices in different chains due to label switching, only the specified chain is considered.

        :param chain_idx: Index of the chain to use for statistics, default is 0
        :return: A summary of the trace in ArviZ format
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been fitted yet.")

        return az.summary(self.trace, coords={"chain": [chain_idx]})

    @staticmethod
    def determine_best_cluster_count(
        data,
        cluster_sizes=None,
        tune=1000,
        samples=500,
        chains=2,
        cores=2,
        eta: float = 2.0,
        sd: float = 1.0,
        mu_prior_std: float = 1.5,
        ic: Literal["waic", "loo"] = "waic",
    ) -> pd.DataFrame:
        """
        Determines the best number of clusters by fitting models with different cluster counts
        and comparing them using WAIC or LOO.

        :param data: A pandas DataFrame or numpy array with continuous values
        :param cluster_sizes: A list specifying the cluster sizes to evaluate
        :param tune: Number of tuning steps for each model
        :param samples: Number of samples to draw for each model
        :param chains: Number of chains to sample in parallel
        :param cores: Number of cores to use for sampling
        :param eta: LKJ prior parameter for correlation matrices
        :param sd: Standard deviation for the half-normal prior on scale parameters
        :param mu_prior_std: Standard deviation for the normal prior on cluster means
        :param ic: Information criterion to use for comparison ("waic" or "loo")
        :return: A DataFrame with comparisons (from ArviZ compare)
        """
        cluster_sizes = cluster_sizes or [2, 3, 4]

        models = {}
        for i in cluster_sizes:
            model = GMvM(
                clusters=i, tune=tune, samples=samples, chains=chains, cores=cores
            )
            model.fit(data, eta=eta, sd=sd, mu_prior_std=mu_prior_std)
            models[f"{i}_clusters"] = model.trace

        comp = az.compare(models, ic=ic)
        return comp