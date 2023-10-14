

import numpy as np

from conftest import N_USERS, N_FEATURES, N_LABELS
from app.dataset import OPENAI_FEATURE_COLS
from app.reduced_dataset import feature_colnames #, REDUCED_FEATURES

REDUCED_N_FEATURES = len(feature_colnames("pca", 2)) # because the fixture is for pca 2


def test_dataset(ds, reduced_ds):
    # the reduced dataset has all the columns of the original dataset, plus columns for each reduction result:
    N_COLS = len(ds.df.columns) + len(reduced_ds.reduced_cols)
    assert reduced_ds.df.shape == (N_USERS, N_COLS)


def test_reduced_features(reduced_ds):
    assert len(reduced_ds.reduced_cols) == 26


def test_labels_df(ds, reduced_ds):
    # the reduced dataset has the same LABELS as the dataset:
    assert sorted(ds.labels_df.columns.tolist()) == sorted(reduced_ds.labels_df.columns.tolist())
    assert reduced_ds.labels_df.shape == (N_USERS, N_LABELS)


def test_feature_cols(reduced_ds):
    assert reduced_ds.feature_cols == ['pca_2_component_1', 'pca_2_component_2'] # because the fixture is for pca 2


def test_x(reduced_ds):
    assert reduced_ds.x.shape == (N_USERS, REDUCED_N_FEATURES)


def test_x_scaled(reduced_ds):
    assert reduced_ds.x_scaled.shape == (N_USERS, REDUCED_N_FEATURES)

    scaled_vals = reduced_ds.x_scaled.to_numpy().flatten()
    # mean centered:
    assert np.isclose(scaled_vals.mean(), 0)
    # unit variance:
    assert np.isclose(scaled_vals.std(), 1)

    #assert np.isclose(scaled_vals.max(), 6.79286)
    #assert np.isclose(scaled_vals.min(), -7.80073)
