

import numpy as np

N_USERS = 7566
N_EMBEDS = 1536 # number of embeddings returned by openai

def test_dataset(ds):
    assert ds.df.shape == (N_USERS, 1559)

def test_labels(ds):
    assert ds.labels.shape == (N_USERS, 23)

def test_x(ds):
    assert ds.x.shape == (N_USERS, N_EMBEDS)

def test_x_scaled(ds):
    assert ds.x_scaled.shape == (N_USERS, N_EMBEDS)

    scaled_vals = ds.x_scaled.to_numpy().flatten()
    # mean centered:
    assert np.isclose(scaled_vals.mean(), 0)
    # unit variance:
    assert np.isclose(scaled_vals.std(), 1)

    assert np.isclose(scaled_vals.max(), 6.79286)
    assert np.isclose(scaled_vals.min(), -7.80073)
