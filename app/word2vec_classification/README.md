## Word2Vec Classification

Save word2vec embeddings dataset with original user labels:

```sh
python -m app.word2vec_classification.dataset
```

Perform classification using the word2vec embeddings dataset:

```sh
FIG_SAVE=true FIG_SHOW=false python -m app.word2vec_classification.job
```
