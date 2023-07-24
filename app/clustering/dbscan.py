import os

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
from sklearn.cluster import DBSCAN

from app.clustering import ClusteringPipeline, CLUSTERING_RESULTS_DIRPATH


N_CLUSTERS = int(os.getenv("N_CLUSTERS", default="2"))

MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", default="5"))


class DBSCANPipeline(ClusteringPipeline):
    # TODO: override performance with a grid search over the eps parameter

    def __init__(self, ds=None, x_scale=True, n_clusters=N_CLUSTERS, min_samples=MIN_SAMPLES):
        super().__init__(ds=ds, x_scale=x_scale)

        self.n_clusters = n_clusters
        self.min_samples = min_samples

        self.results_filestem = os.path.join(CLUSTERING_RESULTS_DIRPATH, f"dbscan_{self.n_clusters}_clusters_{self.min_samples}_min")

        # eps float, default=0.5
        # ... The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        # ... This is not a maximum bound on the distances of points within a cluster.
        # ... This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
        #
        # min_samples int, default=5
        # ... The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
        #
        # metricstr, or callable, default="euclidean"
        # ... The metric to use when calculating distance between instances in a feature array. If metric is a string or callable, it must be one of the options allowed by sklearn.metrics.pairwise_distances for its metric parameter. If metric is "precomputed", X is assumed to be a distance matrix and must be square. X may be a sparse graph, in which case only "nonzero" elements may be considered neighbors for DBSCAN.
        #
        # algorithm{"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
        #
        # leaf_size int, default=30
        # ... Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
        #
        # p float, default=None
        # ... The power of the Minkowski metric to be used to calculate distance between points. If None, then p=2 (equivalent to the Euclidean distance).

        self.model = DBSCAN(eps=0.5, min_samples=self.min_samples, metric="euclidean", algorithm="auto", n_jobs=-1)





if __name__ == "__main__":


    pipeline = DBSCANPipeline()
    pipeline.perform()
