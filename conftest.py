import os
from pytest import fixture

from app.dataset import Dataset

CI_ENV = bool(os.getenv("CI")=="true")

N_USERS = 7566
N_FEATURES = 1536 # number of embeddings returned by openai


@fixture(scope="module")
def ds():
    dataset = Dataset()
    dataset.df
    return dataset

@fixture(scope="module")
def df(ds):
    return ds.df
