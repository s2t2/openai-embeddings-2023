

import os
import json
from abc import ABC

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
from sklearn.cluster import DBSCAN
from pandas import Series, DataFrame

from app.dataset import Dataset
from app.clustering import CLUSTERING_RESULTS_DIRPATH

from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)





class ClusteringPipeline(ABC):

    def __init__(self, ds=None, x_scale=True):
        self.ds = ds or Dataset()
        self.x_scale = bool(x_scale)

        self.x = self.ds.x_scaled if self.x_scale else self.ds.x
        self.labels_df = self.ds.labels_slim

        self.model = None # set this in the child class
        self.results_filestem = None # set this in the child class


    def perform(self):
        self.model.fit(self.x)
        self.save_labels_csv()
        self.save_results_json()

    def save_labels_csv(self):
        self.labels_df[f"cluster_label"] = self.model.labels_
        self.labels_df.to_csv(f"{self.results_filestem}.csv", index=False)

    def save_results_json(self):
        with open(f"{self.results_filestem}.json", "w") as json_file:
            json.dump(self.metrics, json_file, indent=4)

    @property
    def base_results(self):
        return {
            "model_type": self.model.__class__.__name__,
            "model_params": self.model.get_params()
        }

    @property
    def base_metrics(self):
        return {} # override in child class to provide model-specific params

    @property
    def metrics(self):
        results = {**self.base_results, **self.base_metrics} # merge dictionaries

        n_clusters = len(set(self.model.labels_)) # using actual number of labels, instead of n_clusters, because HDBSCAN determines its own n_clusters
        results["n_clusters"] = n_clusters

        value_counts = Series(self.model.labels_).value_counts().rename_axis("cluster_label").reset_index(name="count").to_dict("records")
        results["value_counts"] = value_counts

        # METRICS

        metrics = ["adjusted_rand_score", "adjusted_mutual_info_score",
                "homogeneity_score", "completeness_score", "v_measure_score", "fowlkes_mallows_score",
                "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]
        for metric in metrics:
            results[metric] = {}

        #
        # SUPERVISED / EXTRINSIC
        #

        #result["adjusted_rand_score"]["all_features"] = adjusted_rand_score(x, labels)
        #> ValueError: labels_true must be 1D: shape is (7566, 1536)

        if n_clusters == 2:
            label_cols = ["is_bot", "opinion_community", "is_q"]
        elif n_clusters == 4:
            label_cols = ["fourway_label"]
        elif n_clusters == 6:
            label_cols = ["sixway_label"]
        else:
            label_cols = []

        for label_col in label_cols:
            labels_true = self.labels_df[label_col]
            results["adjusted_rand_score"][label_col] = adjusted_rand_score(labels_true, self.model.labels_)
            results["adjusted_mutual_info_score"][label_col] = adjusted_mutual_info_score(labels_true, self.model.labels_)
            results["homogeneity_score"][label_col] = homogeneity_score(labels_true, self.model.labels_)
            results["completeness_score"][label_col] = completeness_score(labels_true, self.model.labels_)
            results["v_measure_score"][label_col] = v_measure_score(labels_true, self.model.labels_)
            results["fowlkes_mallows_score"][label_col] = fowlkes_mallows_score(labels_true, self.model.labels_)

        #
        # UNSUPERVISED / INTRINSIC
        #
        # intrinsic labels based on distance calculations and can't support categorical data
        #

        results["silhouette_score"]["all_features"] = silhouette_score(self.x, self.model.labels_, metric='euclidean')
        results["calinski_harabasz_score"]["all_features"] = calinski_harabasz_score(self.x, self.model.labels_)
        results["davies_bouldin_score"]["all_features"] = davies_bouldin_score(self.x, self.model.labels_)

        if n_clusters == 2:
            for label_col in ["is_bot", "opinion_community", "is_q"]:
                labels_true_df = DataFrame(self.labels_df[label_col])
                results["silhouette_score"][label_col] = silhouette_score(labels_true_df, self.model.labels_, metric='euclidean')
                results["calinski_harabasz_score"][label_col] = calinski_harabasz_score(labels_true_df, self.model.labels_)
                results["davies_bouldin_score"][label_col] = davies_bouldin_score(labels_true_df, self.model.labels_)
        #elif n_clusters == 4:
        #    #> ValueError: could not convert string to float: 'Pro-Trump Bot'
        #    # so we either need to convert the four-way labels to numeric (which might not be methodologically sound)
        #    # or just skip them
        #elif n_clusters == 6:
        #    #> ValueError: could not convert string to float: 'Pro-Trump Bot'
        #    # so we either need to convert the six-way labels to numeric (which might not be methodologically sound)
        #    # or just skip them

        return results






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
