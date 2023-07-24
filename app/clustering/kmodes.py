import os

# https://github.com/nicodv/kmodes
from kmodes.kmodes import KModes
#from pandas import DataFrame
#from plotly.express import scatter

from app.clustering import N_CLUSTERS, CLUSTERING_RESULTS_DIRPATH, write_results_json, clustering_metrics

N_ITER = int(os.getenv("N_ITER", default="10"))


if __name__ == "__main__":

    from app.dataset import Dataset

    ds = Dataset()
    x = ds.x_scaled

    n_clusters = N_CLUSTERS
    n_iter = N_ITER
    results_filestem = os.path.join(CLUSTERING_RESULTS_DIRPATH, f"kmodes_{n_clusters}_clusters_{n_iter}_iter")

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

    km = KModes(n_clusters=n_clusters, init='Huang', n_init=n_iter, verbose=1, n_jobs=-1, random_state=99)

    km.fit_predict(x)
    labels = km.labels_

    labels_df = ds.labels_slim
    labels_df[f"kmodes_{n_clusters}_label"] = labels
    labels_df.to_csv(f"{results_filestem}.csv", index=False)

    km_params = km.get_params()
    del km_params['cat_dissim'] # this is a function and not serializable for JSON storage
    result = {
        "kmodes_params": km_params,
        "kmodes_cost": km.cost_
    }
    result = clustering_metrics(x=x, cluster_labels=labels, labels_df=labels_df, result=result)
    write_results_json(result=result, json_filepath=f"{results_filestem}.json")


    ## km.cluster_centroids_.shape #> (2, 1536) row per cluster, column per feature
    ## not sure what the centroids are about. I would expect a centroid per cluster?
    #chart_df = DataFrame(km.cluster_centroids_.T, columns=["cluster_1", "cluster_2"])
    #chart_df["feature_name"] = ds.feature_names
    #fig = scatter(chart_df, x="cluster_1", y="cluster_2", color="feature_name")
    #fig.show()
    ## this is not a thing.
