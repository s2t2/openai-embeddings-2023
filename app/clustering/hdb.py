





import os

# https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
from hdbscan import HDBSCAN

# https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
# if ground truth:

# if no ground truth labels:
# https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html
from sklearn import metrics



MIN_CLUSTER_SIZE = os.getenv("MIN_CLUSTER_SIZE")

if __name__ == "__main__":

    from app.dataset import Dataset
    ds = Dataset()
    x = ds.x_scaled

    metric = "euclidean"
    min_cluster_size = MIN_CLUSTER_SIZE
    if min_cluster_size:
        min_cluster_size = int(min_cluster_size)

    hdbscan_params = {"metric": metric, "min_cluster_size": min_cluster_size}
    hdb = HDBSCAN(**hdbscan_params)

    hdb.fit(x)
    labels = hdb.labels_
    #print(hdbscan.labels_[0:25])
    #print(hdbscan.probabilities_[0:25])

    labels_df = ds.df.copy()
    labels_df["hdbscan_label"] = labels
    labels_df["hdbscan_probability"] = hdb.probabilities_
    labels_df.sort_values(by=["hdbscan_label"], inplace=True)
    #print(labels_df.head())
    print(labels_df["hdbscan_label"].value_counts())
    #csv_filepath = os.path.join(DATA_DIRPATH, f"tags_users_onehot_{CLUSTERING_TAGS_LIMIT}_umap_{CLUSTERING_N_COMPONENTS}_cluster_hdbscan_{metric}.csv")
    #labels_df.to_csv(csv_filepath)



    ## EVALUATION

    print(hdb.cluster_persistence_)
    # score of 1.0 represents a perfectly stable cluster that persists over all distance scales, while a score of 0.0 represents a perfectly ephemeral cluster. These scores can be guage the relative coherence of the clusters output by the algorithm.


    sh_score = metrics.silhouette_score(x, labels, metric='euclidean')
    print("SH SCORE:", sh_score)

    ch_score = metrics.calinski_harabasz_score(x, labels)
    print("CH SCORE:", ch_score) #> 2.5839

    db_score = metrics.davies_bouldin_score(x, labels)
    print("DB SCORE:", db_score) #> 3.5719

    # REPORT ON THE SCORES:
    #result = {
    #    "metric": metric,
    #    "n_clusters": len(labels.unique),
    #    "sh_score": sh_score,
    #    "ch_score": ch_score,
    #    "db_score": db_score
    #}
    #print(result)


    breakpoint()


    #hdb.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)

    #hdb.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
