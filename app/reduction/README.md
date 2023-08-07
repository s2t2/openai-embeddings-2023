
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
