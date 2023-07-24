import os

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.cluster import KMeans

from app.clustering import N_CLUSTERS, CLUSTERING_RESULTS_DIRPATH, write_results_json, clustering_metrics

N_ITER = int(os.getenv("N_ITER", default="10"))


if __name__ == "__main__":

    from app.dataset import Dataset

    ds = Dataset()
    x = ds.x_scaled

    n_clusters = N_CLUSTERS
    n_iter = N_ITER
    results_filestem = os.path.join(CLUSTERING_RESULTS_DIRPATH, f"kmeans_{n_clusters}_clusters_{n_iter}_iter")

    #init{"k-means++", "random"}, callable or array-like of shape (n_clusters, n_features), default="k-means++"
    #   Method for initialization:
    #       "k-means++" : selects initial cluster centroids using sampling based on an empirical probability distribution of the points" contribution to the overall inertia. This technique speeds up convergence. The algorithm implemented is “greedy k-means++”. It differs from the vanilla k-means++ by making several trials at each sampling step and choosing the best centroid among them.
    #       "random": choose n_clusters observations (rows) at random from data for the initial centroids.
    #
    # max_iter default = 300

    km = KMeans(n_clusters=n_clusters, n_init=n_iter, verbose=1, random_state=99)

    km.fit_predict(x)
    labels = km.labels_
    # km.cluster_centers_.shape #> (2, 1536) N_CLUSTERS by N_FEATURES

    labels_df = ds.labels_slim
    labels_df[f"kmeans_{n_clusters}_label"] = labels
    labels_df.to_csv(f"{results_filestem}.csv", index=False)

    result = {
        "kmeans_params": km.get_params(),
        "inertia": km.inertia_, # Sum of squared distances of samples to their closest cluster center, weighted by the sample weights if provided.
    }
    result = clustering_metrics(x=x, cluster_labels=labels, labels_df=labels_df, result=result)
    write_results_json(result=result, json_filepath=f"{results_filestem}.json")
