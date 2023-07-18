import os
from pytest import fixture

from app.dataset import Dataset

CI_ENV = bool(os.getenv("CI")=="true")


@fixture(scope="module")
def ds():
    dataset = Dataset()
    dataset.df
    return dataset

@fixture(scope="module")
def df(ds):
    return ds.df
