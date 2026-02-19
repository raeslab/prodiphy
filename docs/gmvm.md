# GMvM - Gaussian Multivariate Mixture

A Gaussian Multivariate Mixture (GMvM) model is a probabilistic approach for clustering continuous multivariate data by modeling the data as a mixture of multivariate Gaussian distributions. Each cluster is characterized by its own mean vector and covariance matrix, allowing the model to capture different shapes, orientations, and sizes of clusters in the multivariate space. This makes it particularly useful for analyzing complex datasets where clusters may have different characteristics.

## Mathematical formulation

Assume there are $N$ observations, each with $D$ continuous features, and a fixed number of latent clusters $C$.
Define the observed vectors as

$$
\mathbf{x}_i \in \mathbb{R}^D, \quad i \in \{1,\dots,N\}
$$

with mixture parameters

$$
\mathbf{w} = (w_1,\dots,w_C), \quad \sum_{c=1}^C w_c = 1,
$$

$$
\boldsymbol{\mu}_c \in \mathbb{R}^D, \quad
\Sigma_c \in \mathbb{R}^{D \times D}, \quad \Sigma_c \succ 0.
$$

The implementation in `GMvM.fit()` uses the following priors for each component $c$:

$$
\mathbf{w} \sim \mathrm{Dirichlet}\left(\frac{1}{C}\mathbf{1}_C\right)
$$

$$
\boldsymbol{\mu}_c \sim \mathcal{N}(\mathbf{0}, \mu_{\text{prior\_std}}^2 I_D)
$$

$$
\Sigma_c = L_c L_c^\top,
$$

where $L_c$ is sampled via an LKJ-Cholesky covariance prior:

$$
L_c \sim \mathrm{LKJCholeskyCov}(\eta, \mathrm{HalfNormal}(\text{sd})).
$$

Conditioned on the parameters, each observation follows a Gaussian mixture likelihood:

$$
p(\mathbf{x}_i \mid \Theta) = \sum_{c=1}^C w_c\,\mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_c, \Sigma_c),
$$

with

$$
\Theta = \{\mathbf{w}, (\boldsymbol{\mu}_c, \Sigma_c)_{c=1}^C\}.
$$

Inference is performed by MCMC sampling (`pm.sample`) and point summaries are reported with ArviZ (`az.summary`).

For cluster assignment in `get_clusters()`, the implementation evaluates per-observation component logits

$$
\log \pi_{ic} = \log w_c + \log \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_c, \Sigma_c),
$$

then samples a latent categorical index

$$
z_i \sim \mathrm{Categorical}(\mathrm{softmax}(\log \pi_{i1},\dots,\log \pi_{iC})).
$$

Posterior predictive draws of $z_i$ are converted to empirical assignment probabilities,

$$
\hat p_{ic} = \Pr(z_i = c \mid \mathbf{x}_i, \text{posterior draws}),
$$

and the reported cluster is $\arg\max_c \hat p_{ic}$.

## Example Usage

The GMvM class can be used to fit a Gaussian Multivariate Mixture model to a dataset and determine the optimal number of clusters. A single GMvM model can also be used to fit a specific number of clusters and then assign data points to one of those clusters.

### Palmer Penguins Data Example

In the example below, we'll use a dataset similar to the Palmer penguins data with four continuous measurements (bill length, bill depth, flipper length, and body mass) to demonstrate clustering of different penguin species.

```python
from prodiphy import GMvM
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Create synthetic penguin-like data with 3 species
    np.random.seed(42)

    # Species 1: Adelie (smaller, shorter bills)
    adelie = np.random.multivariate_normal(
        mean=[39.0, 18.5, 190.0, 3700.0],
        cov=[[3.0, 0.2, 5.0, 200.0],
             [0.2, 1.0, 2.0, 100.0],
             [5.0, 2.0, 30.0, 500.0],
             [200.0, 100.0, 500.0, 50000.0]],
        size=50
    )

    # Species 2: Chinstrap (medium size, longer bills)
    chinstrap = np.random.multivariate_normal(
        mean=[48.5, 18.4, 195.0, 3700.0],
        cov=[[4.0, -0.1, 8.0, 150.0],
             [-0.1, 1.2, 1.5, 80.0],
             [8.0, 1.5, 25.0, 400.0],
             [150.0, 80.0, 400.0, 45000.0]],
        size=68
    )

    # Species 3: Gentoo (larger, shorter bills, longer flippers)
    gentoo = np.random.multivariate_normal(
        mean=[47.5, 15.0, 217.0, 5000.0],
        cov=[[5.0, 0.3, 12.0, 300.0],
             [0.3, 1.5, 3.0, 120.0],
             [12.0, 3.0, 40.0, 600.0],
             [300.0, 120.0, 600.0, 80000.0]],
        size=124
    )

    # Combine data
    data = np.vstack([adelie, chinstrap, gentoo])
    df = pd.DataFrame(data, columns=['bill_length_mm', 'bill_depth_mm',
                                   'flipper_length_mm', 'body_mass_g'])

    # Standardize the data for better convergence
    scaled_data = (df - df.mean()) / df.std()

    # Fit GMvM model with 3 clusters
    model = GMvM(clusters=3, tune=1000, samples=1000, chains=2, cores=2)
    model.fit(scaled_data, eta=2.0, sd=1.0, mu_prior_std=1.5)

    # Get model statistics
    stats = model.get_stats()
    print("Model Statistics:")
    print(stats)

    # Assign clusters to data points
    clusters = model.get_clusters(scaled_data)
    print("\\nCluster Assignments:")
    print(clusters.head(10))

    # Save results
    stats.to_excel("./penguin_gmvm_stats.xlsx")
    clusters.to_excel("./penguin_gmvm_clusters.xlsx")
```

### Determining the Optimal Number of Clusters

Here we'll compare models with different numbers of clusters to determine the optimal model using information criteria.

```python
from prodiphy import GMvM
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Generate synthetic multivariate data with known clusters
    np.random.seed(42)

    # Create 4 clusters with different centers
    cluster1 = np.random.multivariate_normal([0, 0, 0], [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]], 75)
    cluster2 = np.random.multivariate_normal([5, 0, 0], [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]], 75)
    cluster3 = np.random.multivariate_normal([0, 5, 0], [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]], 75)
    cluster4 = np.random.multivariate_normal([0, 0, 5], [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]], 75)

    X = np.vstack([cluster1, cluster2, cluster3, cluster4])

    df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])

    # Standardize the data manually
    scaled_df = (df - df.mean()) / df.std()

    # Compare models with 2, 3, 4, and 5 clusters
    cluster_sizes = [2, 3, 4, 5]
    comps = GMvM.determine_best_cluster_count(
        scaled_df,
        cluster_sizes=cluster_sizes,
        tune=800,
        samples=600,
        chains=2,
        cores=2,
        eta=2.0,
        sd=1.0,
        mu_prior_std=1.5,
        ic="waic"
    )

    print("Model Comparison Results:")
    print(comps)

    # Plot comparison
    az.plot_compare(comps)
    plt.title("Model Comparison: WAIC Scores")
    plt.tight_layout()
    plt.savefig("./gmvm_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
```

![./img/gmvm_model_comparison.png](Output of the example script where the four clusters are in fact found again)

## Model Output Interpretation

### Model Statistics

The `get_stats()` method returns a summary table with the following key parameters:

- **mu_k**: Mean vectors for each cluster k, representing the center of each cluster in the multivariate space
- **sigma_k**: Covariance parameters for each cluster k, controlling the shape and orientation of each cluster
- **w**: Mixture weights, indicating the relative prevalence of each cluster in the data

### Cluster Assignments

The `get_clusters()` method returns a DataFrame with:

- **C1, C2, C3, ...**: Probability that each data point belongs to each cluster
- **cluster**: The assigned cluster label (highest probability cluster)
- **idx**: Index of the original data point

### Model Selection

The `determine_best_cluster_count()` method uses information criteria (WAIC or LOO) to compare models:

- **Lower WAIC/LOO values** indicate better model fit
- **rank**: Ranking of models (1 = best)
- **weight**: Model weights in ensemble averaging
- **p_waic/p_loo**: Effective number of parameters

## Model Assumptions

- Data follows a mixture of multivariate normal distributions
- Each cluster has its own mean vector and covariance matrix
- Cluster membership is probabilistic rather than deterministic
- The number of clusters should be reasonably small relative to the sample size

## Tips for Usage

1. **Data Preprocessing**: Standardize your data if features have different scales (e.g. using Scikit-Learn's `StandardScaler`)
2. **Convergence**: Use adequate tune and sample sizes for complex high-dimensional data
3. **Model Selection**: Compare multiple cluster sizes using `determine_best_cluster_count()`
4. **Label Switching**: Results may vary between chains due to label switching - use `chain_idx` parameter consistently
