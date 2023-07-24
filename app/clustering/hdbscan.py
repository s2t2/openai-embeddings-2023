

import os
import json
from pprint import pprint

# https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
from hdbscan import HDBSCAN
from sklearn import metrics

from app.clustering import CLUSTERING_RESULTS_DIRPATH, write_results_json, clustering_metrics


MIN_CLUSTER_SIZE = os.getenv("MIN_CLUSTER_SIZE")


if __name__ == "__main__":

    from app.dataset import Dataset
    ds = Dataset()
    x = ds.x_scaled
    print(x.shape)

    metric = "euclidean"
    hdbscan_params = {"metric": metric}
    min_cluster_size = MIN_CLUSTER_SIZE
    if min_cluster_size:
        min_cluster_size = int(min_cluster_size)
        hdbscan_params["min_cluster_size"] = min_cluster_size
    print(hdbscan_params)

    hdb = HDBSCAN(**hdbscan_params)
    print(hdb)

    breakpoint()

    hdb.fit(x)
    labels = hdb.labels_
    #print(hdbscan.labels_[0:25])
    #print(hdbscan.probabilities_[0:25])

    labels_df = ds.labels_slim
    labels_df["hdbscan_label"] = labels
    labels_df["hdbscan_probability"] = hdb.probabilities_
    csv_filename = f"hdbscan_clusters_min_{min_cluster_size}.csv" if min_cluster_size else "hdbscan_clusters.csv"
    csv_filepath = os.path.join(CLUSTERING_RESULTS_DIRPATH, csv_filename)
    labels_df.to_csv(csv_filepath, index=False)

    ## EVALUATION

    #print(hdb.cluster_persistence_)
    # score of 1.0 represents a perfectly stable cluster that persists over all distance scales,
    # ... while a score of 0.0 represents a perfectly ephemeral cluster.
    # ... These scores can be guage the relative coherence of the clusters output by the algorithm.

    result = {
        "hdbscan_params": hdbscan_params,
        "cluster_persistence": list(hdb.cluster_persistence_),
    }
    result = clustering_metrics(x=x, cluster_labels=labels, labels_df=labels_df, result=result)
    json_filename = f"hdbscan_clusters_min_{min_cluster_size}.json" if min_cluster_size else "hdbscan_clusters.json"
    json_filepath = os.path.join(CLUSTERING_RESULTS_DIRPATH, json_filename)
    write_results_json(result=result, json_filepath=json_filepath)


    #hdb.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)
    #hdb.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
