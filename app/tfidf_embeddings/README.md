

## TF-IDF

A simple embeddings method.


### Text Embeddings

Run the pipeline. Saves embeddings to HD5 file because CSV export was taking too long.

```sh
python -m app.tfidf_embeddings.pipeline
```

### Dimensionality Reduction

Perform dimensionality reduction on the resulting word and document embeddings, respectively:

```sh
FIG_SAVE=true FIG_SHOW=false python -m app.tfidf_embeddings.reduction
```

### Classification Job

```sh
FIG_SAVE=true FIG_SHOW=false python -m app.tfidf_embeddings.classification
```

This is taking a while. There are so many columns. We should consider using less features. Perhaps 1500 max to be in line with OpenAI text embeddings.

<hr>


## TF-IDF (Max 1500 Features)

Let's try setting max features limit to help models train faster and data to save easier:

```sh
TFIDF_MAX_FEATURES=1500 python -m app.tfidf_embeddings.pipeline
```

```sh
TFIDF_MAX_FEATURES=1500 FIG_SAVE=true FIG_SHOW=false python -m app.tfidf_embeddings.classification
```
