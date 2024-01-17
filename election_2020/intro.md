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


### Application
BigQuery Service:
```sh
python -m app.BigQuery_service
```

fetch embeddings with OpenAI API
```sh
python -m election_2020.embeddings
```

make and save prediction with models from Cloud, for now use Logistic regression
```sh
python -m election_2020.model_prediction
```

