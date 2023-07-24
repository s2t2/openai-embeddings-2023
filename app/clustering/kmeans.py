import os

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.cluster import KMeans

from app.clustering import ClusteringPipeline, CLUSTERING_RESULTS_DIRPATH

N_CLUSTERS = int(os.getenv("N_CLUSTERS", default="2"))
N_ITER = int(os.getenv("N_ITER", default="10"))


class KMeansPipeline(ClusteringPipeline):

    def __init__(self, ds=None, x_scale=True, n_clusters=N_CLUSTERS, n_iter=N_ITER):
        super().__init__(ds=ds, x_scale=x_scale)

        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.results_filestem = os.path.join(CLUSTERING_RESULTS_DIRPATH, f"kmeans_{self.n_clusters}_clusters_{self.n_iter}_iter")

        #init{"k-means++", "random"}, callable or array-like of shape (n_clusters, n_features), default="k-means++"
        #   Method for initialization:
        #       "k-means++" : selects initial cluster centroids using sampling based on an empirical probability distribution of the points" contribution to the overall inertia. This technique speeds up convergence. The algorithm implemented is “greedy k-means++”. It differs from the vanilla k-means++ by making several trials at each sampling step and choosing the best centroid among them.
        #       "random": choose n_clusters observations (rows) at random from data for the initial centroids.
        #
        # max_iter default = 300
        #
        self.model = KMeans(n_clusters=n_clusters, n_init=n_iter, verbose=1, random_state=99)

    @property
    def results(self):
        # inertia : # Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
        kmeans_results = {"inertia": self.model.inertia_,}
        return {**super().results, **kmeans_results}


if __name__ == "__main__":


    pipeline = KMeansPipeline()
    pipeline.perform()
