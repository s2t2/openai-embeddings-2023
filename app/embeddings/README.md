## Text Embeddings Comparison


### OpenAI

See notebooks.

### Word2Vec

```sh
python -m app.embeddings.word2vec

# WORD2VEC_DESTRUCTIVE=true python -m app.embeddings.word2vec

# FIG_SAVE=true FIG_SHOW=true python -m app.embeddings.word2vec
```

Perform dimensionality reduction on the resulting word and document embeddings, respectively:

```sh
python -m app.embeddings.word2vec_reduction

# FIG_SAVE=true FIG_SHOW=true python -m app.embeddings.word2vec_reduction
```