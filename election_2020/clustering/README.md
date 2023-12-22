### Clustering


K-Means Clustering on full dimensional data:

```sh
N_CLUSTERS=2 python -m app.clustering.kmeans
N_CLUSTERS=4 python -m app.clustering.kmeans
```


K-Modes Clustering on full dimensional data:

```sh
N_CLUSTERS=2 python -m app.clustering.kmodes
N_CLUSTERS=4 python -m app.clustering.kmodes
```

Spectral Clustering on full dimensional data:

```sh
python -m app.clustering.spectral

N_CLUSTERS=2 python -m app.clustering.spectral
N_CLUSTERS=4 python -m app.clustering.spectral
```

DBSCAN Clustering on full dimensional data (note: DBSCAN determines the number of clusters):

```sh
python -m app.clustering.dbscan

MIN_SAMPLES=10 python -m app.clustering.dbscan
```


HDBSCAN Clustering on full dimensional data (note: HDBSCAN determines the number of clusters):

```sh
python -m app.clustering.hdbscan

MIN_CLUSTER_SIZE=10 python -m app.clustering.hdbscan
```
