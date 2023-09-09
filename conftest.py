import os
from pytest import fixture

from app.dataset import Dataset, FEATURE_COLS
from app.reduced_dataset import ReducedDataset

CI_ENV = bool(os.getenv("CI")=="true")

N_USERS = 7566
N_FEATURES = 1536 # len(FEATURE_COLS) number of embeddings returned by openai
#N_LABELS = 36 # number of label columns

@fixture(scope="module")
def ds():
    dataset = Dataset()
    dataset.df
    return dataset

@fixture(scope="module")
def df(ds):
    return ds.df

# REDUCED DATASET

@fixture(scope="module")
def reduced_ds():
    dataset = ReducedDataset(reducer_name="pca", n_components=2)
    dataset.df
    return dataset

@fixture(scope="module")
def reduced_df(reduced_ds):
    return reduced_ds.df
