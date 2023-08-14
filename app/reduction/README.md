
### Dimensionality Reduction

#### Pipeline

Perform PCA (or another reduction method) using specified number of components:

```sh
python -m app.reduction.pipeline

N_COMPONENTS=2 FIG_SHOW=true FIG_SAVE=true python -m app.reduction.pipeline
N_COMPONENTS=3 FIG_SHOW=true FIG_SAVE=true python -m app.reduction.pipeline

# other methods:
REDUCER_TYPE="T-SNE" N_COMPONENTS=2 FIG_SHOW=true FIG_SAVE=true python -m app.reduction.pipeline
REDUCER_TYPE="UMAP" N_COMPONENTS=2 FIG_SHOW=true FIG_SAVE=true python -m app.reduction.pipeline
```

This will save the results (plots and embeddings) to the "results/reduction" dir.

> NOTE: T-SNE gets super slow when N_COMPONENTS >= 4 (see T-SNE docs)


#### PCA Tuner

Use PCA to calculate explained variance for each number of components, up to a specified max (to help understand the ideal number of components to use):

```sh
python -m app.pca_tuner

MAX_COMPONENTS=250 FIG_SHOW=true FIG_SAVE=true python -m app.reduction.pca_tuner
```

#### T-SNE Tuner

Use T-SNE KL divergence metric to find optimal n components:

```sh
MAX_COMPONENTS=10 FIG_SHOW=true FIG_SAVE=true  python -m app.reduction.tsne_tuner
```

> NOTE: T-SNE gets super slow when N_COMPONENTS >= 4 (see T-SNE docs)

#### Reduced Dataset

After performing dimensionality reduction using a variety of combinations of methods and number of components, and after saving the results to CSV files, we are now combining all the results into a single file:

```sh
python -m app.reduced_dataset
```

Copy the resulting dataset from the "results" directory into the "data" directory as "data/text-embedding-ada-002/botometer_sample_openai_tweet_embeddings_reduced_20230813.csv.gz". Also upload to Google Drive.
