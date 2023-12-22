
import os

# https://github.com/nicodv/kmodes
from kmodes.kmodes import KModes

from app.clustering import ClusteringPipeline, CLUSTERING_RESULTS_DIRPATH

N_CLUSTERS = int(os.getenv("N_CLUSTERS", default="2"))
N_ITER = int(os.getenv("N_ITER", default="10"))
N_JOBS = int(os.getenv("N_JOBS", default="-1")) # use -1 for parallel


class KModesPipeline(ClusteringPipeline):

    def __init__(self, ds=None, x_scale=True, n_clusters=N_CLUSTERS, n_iter=N_ITER, n_jobs=N_JOBS):
        super().__init__(ds=ds, x_scale=x_scale)

        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.results_filestem = os.path.join(CLUSTERING_RESULTS_DIRPATH, f"kmodes_{self.n_clusters}_clusters_{self.n_iter}_iter")

        # init : {'Huang', 'Cao', 'random' or an ndarray}, default: 'Cao'
        #     Method for initialization:
        #     'Huang': Method in Huang [1997, 1998]
        #     'Cao': Method in Cao et al. [2009]
        #     'random': choose 'n_clusters' observations (rows) at random from data for the initial centroids.
        #
        # n_jobs : If 1, no parallel. If -1 all CPUs are used
        #
        # n_init : Number of time the k-modes algorithm will be run with different centroid seeds.
        # ... The final results will be the best output of n_init consecutive runs in terms of cost.
        # ... default=10

        self.model = KModes(n_clusters=n_clusters, init='Huang', n_init=n_iter, verbose=1, n_jobs=n_jobs, random_state=99)


    @property
    def results(self):
        kmodes_params = self.model.get_params()
        del kmodes_params['cat_dissim'] # this is a function and not serializable for JSON storage

        kmodes_results = {
            "model_params": kmodes_params,
            "kmodes_cost": self.model.cost_
        }
        return {**super().results, **kmodes_results}






if __name__ == "__main__":


    pipeline = KModesPipeline()
    pipeline.perform()


    ## km.cluster_centroids_.shape #> (2, 1536) row per cluster, column per feature
    ## not sure what the centroids are about. I would expect a centroid per cluster?
    #chart_df = DataFrame(km.cluster_centroids_.T, columns=["cluster_1", "cluster_2"])
    #chart_df["feature_name"] = ds.feature_names
    #fig = scatter(chart_df, x="cluster_1", y="cluster_2", color="feature_name")
    #fig.show()
    ## this is not a thing.
