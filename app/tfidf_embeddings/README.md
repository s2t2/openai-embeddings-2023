

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
