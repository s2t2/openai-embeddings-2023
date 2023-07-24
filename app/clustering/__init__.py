

import os
import json
from abc import ABC
from functools import cached_property
from pprint import pprint

from pandas import Series, DataFrame

from app import RESULTS_DIRPATH
from app.dataset import Dataset

from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)


CLUSTERING_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "clustering")


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
        pprint(self.metrics)
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

    @cached_property
    def metrics(self):
        """
        METRICS

            https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
            https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation


        SUPERVISED (LABELS KNOWN):

            adjusted_rand_score: Perfect labeling is scored 1.0.
                ... Poorly agreeing labels (e.g. independent labelings) have lower scores,
                ... and for the adjusted Rand index the score will be negative or close to zero.


            adjusted adjusted_mutual_info_score: Perfect labeling is scored 1.0.
                ... Bad (e.g. independent labelings) have non-positive scores.

            homogeneity_score: each cluster contains only members of a single class (0 - 1, where higher is better).

            completeness_score: all members of a given class are assigned to the same cluster (0 - 1, where higher is better).

            v_measure_score: harmonic mean between homogeneity and completeness scores.

            fowlkes_mallows_score : The score ranges from 0 to 1. A high value indicates a good similarity between two clusters.


        UNSUPERVISED (LABELS NOT KNOWN):

            silhouette_score: The best value is 1 and the worst value is -1.
                ... Values near 0 indicate overlapping clusters.

            calinski_harabasz_score: higher score relates to a model with better defined clusters.

            davies_bouldin_score: The minimum score is zero, with lower values indicating better clustering.

        """

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
