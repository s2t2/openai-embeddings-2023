
import os
import json
from pprint import pprint

from pandas import DataFrame
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score

from app import RESULTS_DIRPATH


N_CLUSTERS = int(os.getenv("N_CLUSTERS", default="2"))


if __name__ == "__main__":

    from app.dataset import Dataset

    ds = Dataset()

    labels_df = ds.df.copy()
    #labels_df["fourway"] = labels_df["fourway_label"].map({"Anti-Trump Bot":0, "Pro-Trump Bot":1, "Anti-Trump Human":2, "Pro-Trump Human":3, "Q-anon Bot":4, "Q-anon Bot":5})
    #labels_df["sixway"] = labels_df["sixway_label"]# .map({})

    #for n_clusters in [2]: #[2, 4, 6]:
    n_clusters = N_CLUSTERS
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
    x = ds.x
    spectral.fit(x)

    labels = spectral.labels_
    print("LABELS:", labels)
    #labels_df[f"spectral_label"] = labels
    labels_df[f"spectral_{n_clusters}_label"] = labels

    results_dirpath = os.path.join(RESULTS_DIRPATH, "clustering")
    csv_filepath = os.path.join(results_dirpath, f"spectral_{n_clusters}_clusters.csv")
    labels_df.to_csv(csv_filepath, index=False)

    print("---------------------")
    print("SUPERVISED METRICS...")

    label_counts = labels_df[f"spectral_{n_clusters}_label"].value_counts().rename_axis("cluster_label").reset_index(name="count").to_dict("records")

    result = {
        "spectral_params": spectral.get_params(),
        "n_clusters": n_clusters, # len(set(labels)),
        "value_counts": label_counts,
        "adjusted_rand_score": {},
        "silhouette_score": {},
        "calinski_harabasz_score": {},
        "davies_bouldin_score": {},
    }


    #print("---------------------")
    #ar = adjusted_rand_score(x, labels)
    #print("ADJ RAND:", ar) #> ValueError: labels_true must be 1D: shape is (7566, 1536)
    if n_clusters == 2:
        ar_bot = adjusted_rand_score(labels_df["is_bot"], labels)
        #print("ADJ RAND BOT:", ar_bot)
        ar_opinion = adjusted_rand_score(labels_df["opinion_community"], labels)
        #print("ADJ RAND OPINION:", ar_opinion)
        result["adjusted_rand_score"]["is_bot"] = ar_bot
        result["adjusted_rand_score"]["opinion_community"] = ar_opinion
    elif n_clusters == 4:
        ar_fourway = adjusted_rand_score(labels_df["fourway_label"], labels)
        result["adjusted_rand_score"]["fourway_label"] = ar_fourway
    elif n_clusters == 6:
        ar_sixway = adjusted_rand_score(labels_df["sixway_label"], labels)
        result["adjusted_rand_score"]["sixway_label"] = ar_sixway



    print("---------------------")
    print("UNSUPERVISED METRICS...")
    # THESE ARE DISTANCE METRICS WHICH DON'T APPLY FOR CATEGORICAL LABELS

    #print("---------------------")
    sh = silhouette_score(x, labels, metric='euclidean')
    result["silhouette_score"]["all_features"] = sh
    #print("SILHOUETTE:", sh)
    if n_clusters == 2:
        sh_bot = silhouette_score(DataFrame(labels_df["is_bot"]), labels, metric='euclidean')
        #print("SILHOUETTE BOT:", sh_bot)
        sh_opinion = silhouette_score(DataFrame(labels_df["opinion_community"]), labels, metric='euclidean')
        #print("SILHOUETTE OPINION:", sh_opinion)
        result["silhouette_score"]["is_bot"] = sh_bot
        result["silhouette_score"]["opinion_community"] = sh_opinion
    #elif n_clusters == 4:
    #    sh_fourway = silhouette_score(DataFrame(labels_df["fourway_label"]), spectral.labels_, metric='precomputed')
    #    #> ValueError: could not convert string to float: 'Pro-Trump Bot'
    #elif n_clusters == 6:
    #    sh_sixway = silhouette_score(DataFrame(labels_df["sixway"]), spectral.labels_, metric='euclidean')
    #    print("FOUR WAY:", sh_sixway)

    #print("---------------------")
    ch_score = calinski_harabasz_score(x, labels)
    #print("CH:", ch_score)
    result["calinski_harabasz_score"]["all_features"] = ch_score
    if n_clusters == 2:
        ch_bot = calinski_harabasz_score(DataFrame(labels_df["is_bot"]), labels)
        #print("CH BOT:", sh_bot)
        ch_opinion = calinski_harabasz_score(DataFrame(labels_df["opinion_community"]), labels)
        #print("CH OPINION:", sh_opinion)
        result["calinski_harabasz_score"]["is_bot"] = ch_bot
        result["calinski_harabasz_score"]["opinion_community"] = ch_opinion
    #elif n_clusters == 4:
    #    ch_fourway = calinski_harabasz_score(DataFrame(labels_df["fourway_label"]), labels)
    #    #> ValueError: could not convert string to float: 'Pro-Trump Bot'
    #    result["calinski_harabasz_score"]["fourway_label"] = ch_fourway

    #print("---------------------")
    db_score = davies_bouldin_score(x, labels)
    #print("DB:", db_score)
    result["davies_bouldin_score"]["all_features"] = db_score
    if n_clusters == 2:
        db_bot = davies_bouldin_score(DataFrame(labels_df["is_bot"]), labels)
        #print("DB BOT:", db_bot)
        db_opinion = davies_bouldin_score(DataFrame(labels_df["opinion_community"]), labels)
        #print("DB OPINION:", db_opinion)
        result["davies_bouldin_score"]["is_bot"] = db_bot
        result["davies_bouldin_score"]["opinion_community"] = db_opinion
    #elif n_clusters == 4:
    #    db_fourway = davies_bouldin_score(DataFrame(labels_df["fourway_label"]), labels)
    #    #>ValueError: could not convert string to float: 'Pro-Trump Bot'
    #    result["davies_bouldin_score"]["fourway_label"] = db_fourway





    pprint(result)

    # SAVE RESULTS TO FILE:
    json_filepath = os.path.join(results_dirpath, f"spectral_{n_clusters}_clusters.json")

    with open(json_filepath, "w") as json_file:
        json.dump(result, json_file, indent=4)
