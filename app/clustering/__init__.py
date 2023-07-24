

import os
import json

from pandas import Series, DataFrame
from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

from app import RESULTS_DIRPATH

CLUSTERING_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "clustering")

N_CLUSTERS = int(os.getenv("N_CLUSTERS", default="2"))


def write_results_json(result:dict, json_filepath:str):
    with open(json_filepath, "w") as json_file:
        json.dump(result, json_file, indent=4)






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


def clustering_metrics(x, cluster_labels, labels_df, result=None):
    """
        x : all features used to obtained the clusters
        labels_pred : y_pred
        labels_df : y_true DataFrame with multiple potential columns to use for labeling, depending on the number of clusters
    """

    n_clusters = len(set(cluster_labels))
    result["n_clusters"] = n_clusters

    label_counts = Series(cluster_labels).value_counts().rename_axis("cluster_label").reset_index(name="count").to_dict("records")
    result["value_counts"] = label_counts
    #print(label_counts)

    # METRICS

    result = result or {}
    metrics = ["adjusted_rand_score", "adjusted_mutual_info_score",
               "homogeneity_score", "completeness_score", "v_measure_score", "fowlkes_mallows_score",
               "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]
    for metric in metrics:
        result[metric] = {}

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
        result["adjusted_rand_score"][label_col] = adjusted_rand_score(labels_df[label_col], cluster_labels)
        result["adjusted_mutual_info_score"][label_col] = adjusted_mutual_info_score(labels_df[label_col], cluster_labels)
        result["homogeneity_score"][label_col] = homogeneity_score(labels_df[label_col], cluster_labels)
        result["completeness_score"][label_col] = completeness_score(labels_df[label_col], cluster_labels)
        result["v_measure_score"][label_col] = v_measure_score(labels_df[label_col], cluster_labels)
        result["fowlkes_mallows_score"][label_col] = fowlkes_mallows_score(labels_df[label_col], cluster_labels)

    #
    # UNSUPERVISED / INTRINSIC
    #
    # intrinsic labels based on distance calculations and can't support categorical data
    #> ValueError: could not convert string to float: 'Pro-Trump Bot'
    # so we either need to convert the four-way and six-way labels to numeric (which might not be methodologically sound)
    # or just skip them

    result["silhouette_score"]["all_features"] = silhouette_score(x, cluster_labels, metric='euclidean')
    result["calinski_harabasz_score"]["all_features"] = calinski_harabasz_score(x, cluster_labels)
    result["davies_bouldin_score"]["all_features"] = davies_bouldin_score(x, cluster_labels)

    if n_clusters == 2:
        for label_col in ["is_bot", "opinion_community", "is_q"]:
            labels_col_as_df = DataFrame(labels_df[label_col])
            result["silhouette_score"][label_col] = silhouette_score(labels_col_as_df, cluster_labels, metric='euclidean')
            result["calinski_harabasz_score"][label_col] = calinski_harabasz_score(labels_col_as_df, cluster_labels)
            result["davies_bouldin_score"][label_col] = davies_bouldin_score(labels_col_as_df, cluster_labels)

    return result
