

import os

# https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
from hdbscan import HDBSCAN

from app.clustering import ClusteringPipeline, CLUSTERING_RESULTS_DIRPATH


MIN_CLUSTER_SIZE = os.getenv("MIN_CLUSTER_SIZE")


class HDBSCANPipeline(ClusteringPipeline):

    def __init__(self, ds=None, x_scale=True, min_cluster_size=MIN_CLUSTER_SIZE):
        super().__init__(ds=ds, x_scale=x_scale)

        hdbscan_params = {"metric": "euclidean"}
        if min_cluster_size:
            min_cluster_size = int(min_cluster_size)
            hdbscan_params["min_cluster_size"] = min_cluster_size
        self.min_cluster_size = min_cluster_size

        filestem = f"hdbscan_clusters_min_{min_cluster_size}" if self.min_cluster_size else "hdbscan_clusters.csv"
        self.results_filestem = os.path.join(CLUSTERING_RESULTS_DIRPATH, filestem)

        self.model = HDBSCAN(**hdbscan_params)


    def save_labels_csv(self):
        self.labels_df["cluster_probability"] = self.model.probabilities_ # hdbscan gives us probabilities as well
        super().save_labels_csv()

    @property
    def results(self):
        # cluster_persistence: score of 1.0 represents a perfectly stable cluster that persists over all distance scales,
        # ... while a score of 0.0 represents a perfectly ephemeral cluster.
        # ... These scores can be guage the relative coherence of the clusters output by the algorithm.
        hdbscan_results = {
            "cluster_persistence": list(self.model.cluster_persistence_),
        }
        return {**super().results, **hdbscan_results}




if __name__ == "__main__":

    pipeline = HDBSCANPipeline()
    pipeline.perform()

    #model.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)
    #model.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
