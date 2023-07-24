
import os
import json
from pprint import pprint

from pandas import DataFrame
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score

from app.clustering import N_CLUSTERS, CLUSTERING_RESULTS_DIRPATH, write_results_json, clustering_metrics



if __name__ == "__main__":

    from app.dataset import Dataset

    ds = Dataset()
    x = ds.x

    n_clusters = N_CLUSTERS
    results_filestem = os.path.join(CLUSTERING_RESULTS_DIRPATH, f"spectral_{n_clusters}_clusters")

    spectral = SpectralClustering(n_clusters=n_clusters,
        #n_components=None,

        #eigen_solver=None,
        random_state=99,

        #n_init=10,
        #gamma=1.0,
        #affinity='rbf',
        #n_neighbors=10,
        #eigen_tol='auto',

        #assign_labels='kmeans',
        #degree=3,
        #coef0=1,
        #kernel_params=None,

        #n_jobs=None,
        verbose=False
    )

    #spectral.fit(ds.x_scaled) #> ConvergenceWarning: Number of distinct clusters (1) found smaller than n_clusters (2). Possibly due to duplicate points in X.
    spectral.fit(x)
    labels = spectral.labels_

    labels_df = ds.labels_slim
    labels_df[f"spectral_{n_clusters}_label"] = labels
    labels_df.to_csv(f"{results_filestem}.csv", index=False)

    result = {"spectral_params": spectral.get_params()}
    result = clustering_metrics(x=x, cluster_labels=labels, labels_df=labels_df, result=result)
    write_results_json(result=result, json_filepath=f"{results_filestem}.json")
