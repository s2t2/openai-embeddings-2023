# openai-embeddings-2023

Dimensionality Reduction on Twitter Data using OpenAI Embeddings


Can we use ChatGPT's embeddings to reproduce our previous research?

Can ChatGPT discern bot status, political sentiment, and q-anon support, based on user profiles and tweets?

## Preliminary Results

For a balanced sample of 300 users and a small sample of their tweets, it looks like the ChatGPT embeddings of their tweets might be able to be used to discern bot status:

<a href="https://s2t2.github.io/openai-embeddings-2023/reduction_results/tweets/bot_label/pca_2.html">

![newplot (2)](https://github.com/s2t2/openai-embeddings-2023/assets/1328807/751d5933-869a-4f05-8b86-d5283c95dfff)

</a>


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
python -m app.reduction_pipeline

N_COMPONENTS=2 FIG_SHOW=true FIG_SAVE=true python -m app.reduction_pipeline
N_COMPONENTS=3 FIG_SHOW=true FIG_SAVE=true python -m app.reduction_pipeline

# other methods:
REDUCER_TYPE="T-SNE" python -m app.reduction_pipeline
REDUCER_TYPE="UMAP" python -m app.reduction_pipeline
```


Use PCA to calculate explained variance for each number of components, up to a specified max (to help understand the ideal number of components to use):

```sh
python -m app.pca_tuner

MAX_COMPONENTS=250 FIG_SHOW=true FIG_SAVE=true python -m app.pca_tuner
```




## Testing

```sh
pytest
```
