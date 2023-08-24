

from numpy import nan, isclose

from conftest import N_USERS, N_FEATURES, N_LABELS


def test_dataset(ds):
    n_cols = N_FEATURES + N_LABELS
    assert ds.df.shape == (N_USERS, n_cols)


def test_labels(ds):
    assert ds.labels.shape == (N_USERS, N_LABELS) # 32 label cols

    assert ds.df["fourway_label"].value_counts().to_dict() == {
        'Anti-Trump Human': 3010, 'Anti-Trump Bot': 1881,
        'Pro-Trump Human': 1456, 'Pro-Trump Bot': 1219
    }


def test_x(ds):
    assert ds.x.shape == (N_USERS, N_FEATURES)


def test_x_scaled(ds):
    assert ds.x_scaled.shape == (N_USERS, N_FEATURES)

    scaled_vals = ds.x_scaled.to_numpy().flatten()
    # mean centered:
    assert isclose(scaled_vals.mean(), 0)
    # unit variance:
    assert isclose(scaled_vals.std(), 1)

    assert isclose(scaled_vals.max(), 6.79286)
    assert isclose(scaled_vals.min(), -7.80073)


def test_score_thresholding(ds):

    #breakpoint()

    assert ds.df["avg_toxicity"].isna().sum() == 0

    # NEWS QUALITY
    # some of the scores are null. we need to consider imputing them
    #assert ds.df["is_factual"].value_counts(dropna=False).to_dict() == {nan: 4274, False: 1696, True: 1596}

    assert ds.df["is_factual"].isna().sum() == 4274
    assert ds.df["is_factual"].notna().sum() == 3292
    # there are a significant number of missing values
