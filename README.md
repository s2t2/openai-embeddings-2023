# openai-embeddings-2023

Dimensionality Reduction on Twitter Data using OpenAI Embeddings

## Research Questions

Can we use ChatGPT's embeddings to reproduce our previous research?

Can ChatGPT discern bot status, political sentiment, and q-anon support, based on user profiles and tweets?


## Setup

Create and/or activate virtual environment:

```sh
conda create -n openai-env python=3.10
conda activate openai-env
```

Install package dependencies:

```sh
pip install -r requirements.txt
```

Obtain a copy of the "botometer_sample_openai_tweet_embeddings_20230704.csv.gz" CSV file, and store it in the "data/text-embedding-ada-002" directory in this repo. This file is ignored from version control.

## Usage

### Dataset Loading

Demonstrate ability to load the dataset:

```sh
python -m app.dataset
```

### Dimensionality Reduction

Perform PCA (or another reduction method) using specified number of components:

```sh
python -m app.reduction.pipeline

N_COMPONENTS=2 FIG_SHOW=true FIG_SAVE=true python -m app.reduction.pipeline
N_COMPONENTS=3 FIG_SHOW=true FIG_SAVE=true python -m app.reduction.pipeline

# other methods:
REDUCER_TYPE="T-SNE" N_COMPONENTS=2 FIG_SHOW=true FIG_SAVE=true python -m app.reduction.pipeline
REDUCER_TYPE="UMAP" N_COMPONENTS=2 FIG_SHOW=true FIG_SAVE=true python -m app.reduction.pipeline
```


Use PCA to calculate explained variance for each number of components, up to a specified max (to help understand the ideal number of components to use):

```sh
python -m app.pca_tuner

MAX_COMPONENTS=250 FIG_SHOW=true FIG_SAVE=true python -m app.reduction.pca_tuner
```

Use T-SNE KL divergence metric to find optimal n components:

```sh
MAX_COMPONENTS=10 FIG_SHOW=true FIG_SAVE=true  python -m app.reduction.tsne_tuner
```

### Clustering

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

DBSCAN Clustering on full dimensional data:

```sh
python -m app.clustering.dbscan

N_CLUSTERS=2 MIN_SAMPLES=10 python -m app.clustering.dbscan
N_CLUSTERS=4 MIN_SAMPLES=10 python -m app.clustering.dbscan
```


HDBSCAN Clustering on full dimensional data (note: HDBSCAN determines the number of clusters):

```sh
python -m app.clustering.hdbscan

MIN_CLUSTER_SIZE=50 python -m app.clustering.hdbscan
```

## Testing

```sh
pytest
```
