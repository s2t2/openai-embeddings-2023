
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
    labels_df[f"spectral_{n_clusters}"] = labels

    results_dirpath = os.path.join(RESULTS_DIRPATH, "clustering")
    csv_filepath = f"spectral_labels_{N_CLUSTERS}_clusters.csv"
    labels_df.to_csv(csv_filepath, index=False)

    print("---------------------")
    print("SUPERVISED METRICS...")

    print("---------------------")
    #ar = adjusted_rand_score(x, labels)
    #print("ADJ RAND:", ar) #> ValueError: labels_true must be 1D: shape is (7566, 1536)
    if n_clusters == 2:
        ar_bot = adjusted_rand_score(labels_df["is_bot"], labels)
        print("ADJ RAND BOT:", ar_bot)

        ar_opinion = adjusted_rand_score(labels_df["opinion_community"], labels)
        print("ADJ RAND OPINION:", ar_opinion)



    print("---------------------")
    print("UNSUPERVISED METRICS...")


    print("---------------------")
    sh = silhouette_score(x, labels, metric='euclidean')
    print("SILHOUETTE:", sh)
    if n_clusters == 2:
        sh_bot = silhouette_score(DataFrame(labels_df["is_bot"]), labels, metric='euclidean')
        print("SILHOUETTE BOT:", sh_bot)

        sh_opinion = silhouette_score(DataFrame(labels_df["opinion_community"]), labels, metric='euclidean')
        print("SILHOUETTE OPINION:", sh_opinion)
    #elif n_clusters == 4:
    #    sh_fourway = silhouette_score(DataFrame(labels_df["fourway"]), spectral.labels_, metric='euclidean')
    #    print("FOUR WAY:", sh_fourway)
    #elif n_clusters == 6:
    #    sh_sixway = silhouette_score(DataFrame(labels_df["sixway"]), spectral.labels_, metric='euclidean')
    #    print("FOUR WAY:", sh_sixway)

    print("---------------------")
    ch_score = calinski_harabasz_score(x, labels)
    print("CH:", ch_score)
    if n_clusters == 2:
        ch_bot = calinski_harabasz_score(DataFrame(labels_df["is_bot"]), labels)
        print("CH BOT:", sh_bot)

        ch_opinion = calinski_harabasz_score(DataFrame(labels_df["opinion_community"]), labels)
        print("CH OPINION:", sh_opinion)

    print("---------------------")
    db_score = davies_bouldin_score(x, labels)
    print("DB:", db_score)
    if n_clusters == 2:
        db_bot = davies_bouldin_score(DataFrame(labels_df["is_bot"]), labels)
        print("DB BOT:", db_bot)

        db_opinion = davies_bouldin_score(DataFrame(labels_df["opinion_community"]), labels)
        print("DB OPINION:", db_opinion)
