

import numpy as np

from conftest import N_USERS, N_FEATURES, N_LABELS
from app.reduced_dataset import feature_colnames

REDUCED_N_FEATURES = len(feature_colnames("pca", 2)) # because the fixture is for pca 2


def test_dataset(reduced_ds):
    pca_features = feature_colnames("pca", 2) + feature_colnames("pca", 3) + feature_colnames("pca", 7)
    tsne_features = feature_colnames("tsne", 2) + feature_colnames("tsne", 3) + feature_colnames("tsne", 4)
    umap_features = feature_colnames("umap", 2) + feature_colnames("umap", 3)
    N_REDUCTION_FEATURES = len(pca_features) + len(tsne_features) + len(umap_features)
    N_COLS = N_FEATURES + N_REDUCTION_FEATURES + N_LABELS
    assert reduced_ds.df.shape == (N_USERS, N_COLS)

def test_labels_df(reduced_ds):
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
