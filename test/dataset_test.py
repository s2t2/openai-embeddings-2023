

from numpy import nan, isclose
from pandas import DataFrame

from conftest import N_USERS, N_FEATURES #, N_LABELS
from app.dataset import DATASET_VERSION, CSV_FILEPATH

def test_dataset(ds):
    assert ds.csv_filepath == CSV_FILEPATH # upstream (reading in)
    assert ds.version == DATASET_VERSION # downstream (writing out)
    assert len(ds.feature_cols) == N_FEATURES

def test_df(ds):
    assert isinstance(ds.df, DataFrame)
    assert len(ds.df) == N_USERS

def test_labels_df(ds):
    assert isinstance(ds.labels_df, DataFrame)
    assert len(ds.labels_df) == N_USERS

def test_x(ds):
    assert isinstance(ds.x, DataFrame)
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





def test_custom_labels(ds):
    assert ds.df["fourway_label"].value_counts().to_dict() == {
        'Anti-Trump Human': 3010,
        'Anti-Trump Bot': 1881,
        'Pro-Trump Human': 1456,
        'Pro-Trump Bot': 1219
    }

    assert ds.df["eightway_label"].value_counts().to_dict() == {
        'Anti-Trump Bot High Quality': 1384,
        'Pro-Trump Bot Low Quality': 1105,
        'Anti-Trump Bot Low Quality': 463,
        'Anti-Trump Human High Quality': 171,
        'Pro-Trump Human Low Quality': 74,
        'Anti-Trump Human Low Quality': 54,
        'Pro-Trump Bot High Quality': 33,
        'Pro-Trump Human High Quality': 8 # LOW SUPPORT
    }

    assert ds.df["allway_label"].value_counts().to_dict() == {
        'Anti-Trump Bot Non-toxic High Quality': 1339,
        'Pro-Trump Bot Non-toxic Low Quality': 1073,
        'Anti-Trump Bot Non-toxic Low Quality': 439,
        'Anti-Trump Human Non-toxic High Quality': 130,
        'Pro-Trump Human Non-toxic Low Quality': 66,
        'Anti-Trump Bot Toxic High Quality': 45,
        'Anti-Trump Human Toxic High Quality': 41,
        'Anti-Trump Human Non-toxic Low Quality': 39,
        'Pro-Trump Bot Non-toxic High Quality': 33,
        'Pro-Trump Bot Toxic Low Quality': 32,
        'Anti-Trump Bot Toxic Low Quality': 24,
        'Anti-Trump Human Toxic Low Quality': 15,
        'Pro-Trump Human Toxic Low Quality': 8, # LOW SUPPORT
        'Pro-Trump Human Non-toxic High Quality': 7, # LOW SUPPORT
        'Pro-Trump Human Toxic High Quality': 1 #  LOW SUPPORT
    }




def test_score_thresholding(ds):

    #breakpoint()

    assert ds.df["avg_toxicity"].isna().sum() == 0

    # NEWS QUALITY
    # some of the scores are null. we need to consider imputing them
    #assert ds.df["is_factual"].value_counts(dropna=False).to_dict() == {nan: 4274, False: 1696, True: 1596}

    assert ds.df["is_factual"].isna().sum() == 4274
    assert ds.df["is_factual"].notna().sum() == 3292
    # there are a significant number of missing values
