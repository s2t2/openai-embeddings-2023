# openai-embeddings-2023

OpenAI Text Embeddings for User Classification in Social Networks

  + [Results Website](https://s2t2.github.io/openai-embeddings-2023/index.html)
  + [Conference Talk (INFORMS 2023)](https://www.youtube.com/watch?v=AmF-5D4p1_4)


## Setup

### Virtual Environment

Create and/or activate virtual environment:

```sh
conda create -n openai-env python=3.10
conda activate openai-env
```

Install package dependencies:

```sh
pip install -r requirements.txt
```

### OpenAI API

Obtain an OpenAI API Key (i.e. `OPENAI_API_KEY`). We initially fetched embeddings from the OpenAI API via the [notebooks](/notebooks/README.md), but the service code has been re-implemented here afterwards, in case you want to experiment with obtaining your own embeddings.

### Users Sample

Obtain a copy of the "botometer_sample_openai_tweet_embeddings_20230724.csv.gz" CSV file, and store it in the "data/text-embedding-ada-002" directory in this repo. This file was generated by the [notebooks](/notebooks/README.md), and is ignored from version control because it contains user identifiers.

### Cloud Storage

We are saving trained models to Google Cloud Storage. You will need to create a project on Google Cloud, and enable the Cloud Storage API as necessary. Then create a service account and download the service account JSON credentials file, and store it in the root directory, called "google-credentials.json". This file has been ignored from version control.

From the cloud storage console, create a new bucket, and note its name (i.e. `BUCKET_NAME`).


### Environment Variables

Create a local ".env" file and add contents like the following:

```sh
# this is the ".env" file...

OPENAI_API_KEY="sk__________"

GOOGLE_APPLICATION_CREDENTIALS="/path/to/openai-embeddings-2023/google-credentials.json"
BUCKET_NAME="my-bucket"
```

## Usage

### OpenAI Service

Fetch some example embeddings from OpenAI API:

```sh
python -m app.openai_service
```


### Dataset Loading

Demonstrate ability to load the dataset:

```sh
python -m app.dataset
```

### Data Analysis

Perform machine learning and other analyses on the data:

OpenAI Embeddings:

  + [Dimensionality Reduction](app/reduction/README.md)
  + [Clustering](app/clustering/README.md)
  + [Classification](app/classification/README.md)
  + [Reduced Classification](app/reduced_classification/README.md)

Word2Vec Embeddings:

  + [Dimensionality Reduction](app/word2vec_embeddings/README.md)
  + [Classification](app/word2vec_classification/README.md)


## Testing

```sh
pytest --disable-warnings
```
