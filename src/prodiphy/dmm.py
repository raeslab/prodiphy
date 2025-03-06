import pymc as pm
import pandas as pd
import numpy as np
import arviz as az


class DMM:
    def __init__(self, clusters:int, chains=4, cores=4, samples=1000, tune=1500):
        """
        Initializes the DMM class with the number clusters to include as well as the desired
        number of chains and cores for sampling.

        :param clusters: Number of clusters to include in the model
        :param chains: Number of chains to sample in parallel, default is 4
        :param cores: Number of cores to use for sampling, default is 4
        :param samples: Number of samples to draw, default is 1000
        :param tune: Number of tuning steps, default is 1500
        """
        self.clusters = clusters
        self.model = None
        self.trace = None
        self.chains = chains
        self.cores = cores
        self.samples = samples
        self.tune = tune

    def fit(self, data: pd.DataFrame, lower=1, upper=500, priors=None, weights=None):
        """
        Fits a Dirichlet Multinomial Mixture model to the input data.

        :param data: A pandas DataFrame with integer counts
        :param lower: Lower bound for the concentration parameter
        :param upper: Upper bound for the concentration parameter
        :param priors: Prior probabilities for the features within each cluster (shape = n_clusters, n_features)
        :param weights: Prior probabilities for the clusters' prevalence
        """

        if data.empty:
            raise ValueError("Input data is empty.")

        n_reads = int(np.median(data.sum(axis=1)))

        print(f"Median reads: {n_reads}")

        n_features = data.shape[1]
        if priors is None:
            priors = data.mean(axis=0) / n_reads

        if weights is None:
            weights = np.ones(self.clusters) / self.clusters

        with pm.Model() as self.model:
            # Create the weights for each cluster in the mixture model
            w = pm.Dirichlet("w", weights)

            # Dirichlet coefficients
            p = pm.Dirichlet("p", priors, shape=(self.clusters, n_features))
            conc = pm.Uniform("conc", lower=lower, upper=upper, shape=(self.clusters,))

            for c in range(self.clusters):
                _ = pm.Deterministic(f"a[{c}]", conc[c] * p[c])

            # Mixture is created here
            Multinomials = [
                pm.DirichletMultinomial.dist(n_reads, a=p[k] * conc[k], shape=n_features)
                for k in range(self.clusters)
            ]

            # Likelihood
            obs = pm.Mixture("obs", w=w, comp_dists=Multinomials, observed=data)

            # Run trace
            self.trace = pm.sample(self.samples, cores=self.cores, tune=self.tune, chains=self.chains, target_accept=0.90, idata_kwargs={"log_likelihood": True})

    def get_stats(self):
        """
        Returns a summary of the trace in Arviz format. As different clusters may have different indices in different
        chains, only the first chain is considered. This however makes it impossible to compute rhat values.

        :return: A summary of the trace in Arviz format
        """

        if self.model is None or self.trace is None:
            raise ValueError("Model has not been fitted yet.")

        summary_df = az.summary(self.trace, coords={"chain": [0]})

        return summary_df