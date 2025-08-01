import pymc as pm
import pandas as pd
import numpy as np
import arviz as az

from typing import Literal


class DMM:
    def __init__(self, clusters: int, chains=4, cores=4, samples=1000, tune=1500):
        """
        Initializes the DMM class with the number clusters to include as well as the desired
        number of chains and cores for sampling.

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

    def _prepare_data(self, data, priors, weights):
        """Prepares the necessary statistics and priors for the model."""
        if data.empty:
            raise ValueError("Input data is empty.")

        n_reads = int(np.median(data.sum(axis=1)))
        n_features = data.shape[1]

        if priors is None:
            priors = data.mean(axis=0) / n_reads
        if weights is None:
            weights = np.ones(self.clusters) / self.clusters

        return n_reads, n_features, priors, weights

    def _build_model_components(
        self, n_reads, n_features, priors, weights, conc_bounds
    ):
        """Builds the common components of the PyMC model as these are needed twice, once for fitting and once for recovery."""
        lower, upper = conc_bounds

        with pm.Model() as model:
            w = pm.Dirichlet("w", weights)
            p = pm.Dirichlet("p", priors, shape=(self.clusters, n_features))
            conc = pm.Uniform("conc", lower=lower, upper=upper, shape=(self.clusters,))

            for c in range(self.clusters):
                pm.Deterministic(f"a[{c}]", conc[c] * p[c])

            components = [
                pm.DirichletMultinomial.dist(
                    n_reads, a=p[k] * conc[k], shape=n_features
                )
                for k in range(self.clusters)
            ]

        return model, w, p, conc, components

    def fit(self, data: pd.DataFrame, lower=1, upper=500, priors=None, weights=None, sample_kwargs=None):
        """
        Fits a Dirichlet Multinomial Mixture model to the input data.

        :param data: A pandas DataFrame with integer counts
        :param lower: Lower bound for the concentration parameter
        :param upper: Upper bound for the concentration parameter
        :param priors: Prior probabilities for the features within each cluster (shape = n_clusters, n_features)
        :param weights: Prior probabilities for the clusters' prevalence
        :param sample_kwargs: Additional keyword arguments to pass to pm.sample()
        """
        n_reads, n_features, priors, weights = self._prepare_data(data, priors, weights)

        if sample_kwargs is None:
            sample_kwargs = {}

        with pm.Model() as self.model:
            _, w, p, conc, components = self._build_model_components(
                n_reads, n_features, priors, weights, conc_bounds=(lower, upper)
            )

            obs = pm.Mixture("obs", w=w, comp_dists=components, observed=data)

            self.trace = pm.sample(
                self.samples,
                cores=self.cores,
                tune=self.tune,
                chains=self.chains,
                target_accept=0.90,
                idata_kwargs={"log_likelihood": True},
                **sample_kwargs
            )

    def get_clusters(self, data, chain_idx=0):
        """
        Assigns clusters to the input data based on the fitted Dirichlet Multinomial Mixture model.

        This method creates a new model to compute the log probabilities that each data point
        belongs to each component of the mixture model. It then uses the posterior predictive
        distribution to assign clusters to each data point.

        :param data: A pandas DataFrame with integer counts
        :param chain_idx: Index of the chain to use for cluster assignment, default is 0
        :return: A pandas DataFrame with cluster assignments and probabilities for each cluster
        :raises ValueError: If the input data is empty
        """

        if self.model is None or self.trace is None:
            raise ValueError("Model has not been fitted yet.")

        n_reads, n_features, priors, weights = self._prepare_data(data, None, None)

        with pm.Model() as recovery_model:
            _, w, p, conc, components = self._build_model_components(
                n_reads, n_features, priors, weights, conc_bounds=(0, 1000)
            )

            log_probs = pm.math.concatenate(
                [
                    [pm.math.log(w[i]) + pm.logp(components[i], data)]
                    for i in range(self.clusters)
                ],
                axis=0,
            )

            _ = pm.Categorical("idx", logit_p=log_probs.T)
            pp = pm.sample_posterior_predictive(self.trace, var_names=["idx"])
            idx = pp.posterior_predictive["idx"]

            n_draws = idx.shape[1]
            output = []

            for i in range(data.shape[0]):
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
        Returns a summary of the trace in Arviz format. As different clusters may have different indices in different
        chains, only the first chain is considered. This however makes it impossible to compute rhat values.

        :return: A summary of the trace in Arviz format
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model has not been fitted yet.")

        return az.summary(self.trace, coords={"chain": [chain_idx]})

    @staticmethod
    def determine_best_cluster_count(
        df,
        cluster_sizes=None,
        tune=1000,
        samples=500,
        chains=2,
        cores=2,
        lower=10,
        upper=30,
        ic: Literal["waic", "loo"] = "waic",
    ) -> pd.DataFrame:
        """
        Determines the best number of clusters by fitting models with different cluster counts
        and comparing them using WAIC.

        :param df: A pandas DataFrame with integer counts.
        :param cluster_sizes: A list specifying the cluster sizes to evaluate.
        :param tune: Number of tuning steps for each model.
        :param samples: Number of samples to draw for each model.
        :param chains: Number of chains to sample in parallel.
        :param cores: Number of cores to use for sampling.
        :param lower: Lower bound for the concentration parameter.
        :param upper: Upper bound for the concentration parameter.
        :param ic: Information criterion to use for comparison ("waic" or "loo").
        :return: A DataFrame with comparisons (from Arviz compare).
        """

        cluster_sizes = cluster_sizes or [3, 4, 5]

        models = {}
        for i in cluster_sizes:
            model = DMM(
                clusters=i, tune=tune, samples=samples, chains=chains, cores=cores
            )
            model.fit(df, lower=lower, upper=upper)
            models[f"{i}_clusters"] = model.trace

        comp = az.compare(models, ic=ic)
        return comp
