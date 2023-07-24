
import os

from sklearn.cluster import SpectralClustering

from app.clustering import ClusteringPipeline, CLUSTERING_RESULTS_DIRPATH

N_CLUSTERS = int(os.getenv("N_CLUSTERS", default="2"))
N_ITER = int(os.getenv("N_ITER", default="10"))


class SpectralPipeline(ClusteringPipeline):

    def __init__(self, ds=None, x_scale=False, n_clusters=N_CLUSTERS, n_iter=N_ITER):
        super().__init__(ds=ds, x_scale=x_scale)

        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.results_filestem = os.path.join(CLUSTERING_RESULTS_DIRPATH, f"spectral_{self.n_clusters}_clusters_{self.n_iter}_iter")

        self.model = SpectralClustering(n_clusters=self.n_clusters, random_state=99, verbose=False,
            #n_components=None,
            #eigen_solver=None,
            n_init=self.n_iter,
            #gamma=1.0,
            #affinity='rbf',
            #n_neighbors=10,
            #eigen_tol='auto',
            #assign_labels='kmeans',
            #degree=3,
            #coef0=1,
            #kernel_params=None,
            #n_jobs=None,
        )




if __name__ == "__main__":


    pipeline = SpectralPipeline()
    pipeline.perform()
