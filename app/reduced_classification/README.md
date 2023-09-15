### Classification on Reduced Embeddings

First run the reduction pipeline and create the reduced dataset using the reduction results.

Now produce classification results using the reduced embeddings:


```sh
python -m app.reduced_classification.pipeline
```


Single CSV results file for all classification and reduced classification results:


```sh
python -m app.reduced_classification.reporter
```
