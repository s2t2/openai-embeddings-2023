

import numpy as np

from conftest import N_USERS, N_FEATURES


def test_dataset(ds):
    assert ds.df.shape == (N_USERS, 1559)


def test_labels(ds):
    assert ds.labels.shape == (N_USERS, 23)


def test_x(ds):
    assert ds.x.shape == (N_USERS, N_FEATURES)


def test_x_scaled(ds):
    assert ds.x_scaled.shape == (N_USERS, N_FEATURES)

    scaled_vals = ds.x_scaled.to_numpy().flatten()
    # mean centered:
    assert np.isclose(scaled_vals.mean(), 0)
    # unit variance:
    assert np.isclose(scaled_vals.std(), 1)

    assert np.isclose(scaled_vals.max(), 6.79286)
    assert np.isclose(scaled_vals.min(), -7.80073)
