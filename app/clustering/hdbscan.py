

import os
import json
from pprint import pprint

# https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
from hdbscan import HDBSCAN
from sklearn import metrics

from app import RESULTS_DIRPATH


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

    hdb.fit(x)
    labels = hdb.labels_
    #print(hdbscan.labels_[0:25])
    #print(hdbscan.probabilities_[0:25])


    labels_df = ds.labels.copy()
    labels_df["hdbscan_label"] = labels
    labels_df["hdbscan_probability"] = hdb.probabilities_
    labels_df.sort_values(by=["hdbscan_label"], inplace=True)
    #print(labels_df.head())
    print(labels_df["hdbscan_label"].value_counts())

    results_dirpath = os.path.join(RESULTS_DIRPATH, "clustering")
    csv_filename = f"hdbscan_labels_min_{min_cluster_size}.csv" if min_cluster_size else "hdbscan_labels.csv"
    csv_filepath = os.path.join(results_dirpath, csv_filename)
    labels_df.to_csv(csv_filepath, index=False)



    ## EVALUATION

    label_counts = labels_df["hdbscan_label"].value_counts().rename_axis("cluster_label").reset_index(name="count").to_dict("records")

    print(hdb.cluster_persistence_)
    # score of 1.0 represents a perfectly stable cluster that persists over all distance scales, while a score of 0.0 represents a perfectly ephemeral cluster. These scores can be guage the relative coherence of the clusters output by the algorithm.

    sh_metric = "euclidean"
    sh_score = metrics.silhouette_score(x, labels, metric=sh_metric)
    print("SH SCORE:", sh_score)

    ch_score = metrics.calinski_harabasz_score(x, labels)
    print("CH SCORE:", ch_score)

    db_score = metrics.davies_bouldin_score(x, labels)
    print("DB SCORE:", db_score)

    # REPORT ON THE SCORES:
    result = {
        "hdbscan_params": hdbscan_params,
        "n_clusters": len(set(labels)),
        #"cluster_labels": labels,
        #"cluster_probabilities": hdb.probabilities_,
        "value_counts": label_counts,
        "cluster_persistence": list(hdb.cluster_persistence_),
        "silhouette_metric": sh_metric,
        "silhouette_score": sh_score,
        "calinski_harabasz_score": ch_score,
        "davies_bouldin_score": db_score
    }
    pprint(result)

    # SAVE RESULTS TO FILE:
    json_filename = f"hdbscan_results_min_{min_cluster_size}.json" if min_cluster_size else "hdbscan_results.json"
    json_filepath = os.path.join(results_dirpath, json_filename)

    with open(json_filepath, "w") as json_file:
        json.dump(result, json_file, indent=4)




    #hdb.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)

    #hdb.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
