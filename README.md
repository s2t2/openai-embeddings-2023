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

Obtain a copy of the "botometer_sample_openai_tweet_embeddings_20230724.csv.gz" CSV file, and store it in the "data/text-embedding-ada-002" directory in this repo. This file is ignored from version control.

## Usage

### Dataset Loading

Demonstrate ability to load the dataset:

```sh
python -m app.dataset
```

### Data Analysis

Perform machine learning and other analyses on the data:

  + [Dimensionality Reduction](app/reduction/README.md)
  + [Clustering](app/clustering/README.md)
  + [Classification](app/classification/README.md)


## Testing

```sh
pytest --disable-warnings
```
