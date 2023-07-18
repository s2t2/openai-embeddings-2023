




import os
from functools import cached_property

from pandas import read_csv, DataFrame
from sklearn.preprocessing import scale #, StandardScaler

from app import DATA_DIRPATH

CSV_FILEPATH = os.path.join(DATA_DIRPATH, "text-embedding-ada-002", "botometer_sample_openai_tweet_embeddings_20230704.csv.gz")

LABEL_COLS = [
    'user_id', 'created_on', 'screen_name_count', 'screen_names', 'status_count', 'rt_count', 'rt_pct',
    'avg_toxicity', 'avg_fact_score', 'opinion_community', 'is_bot', 'is_q',
    'tweet_texts',
    'bom_cap', 'bom_astroturf', 'bom_fake_follower', 'bom_financial', 'bom_other',
    'opinion_label', 'bot_label', 'q_label', 'group_label'
]


class Dataset():
    def __init__(self, csv_filepath=CSV_FILEPATH, label_cols=LABEL_COLS):
        self.title = "Tweet Embeddings Dataset"
        self.csv_filepath = csv_filepath
        self.label_cols = label_cols

    @cached_property
    def df(self):
        return read_csv(self.csv_filepath)

    @cached_property
    def labels(self):
        return self.df[self.label_cols].copy()

    @cached_property
    def x(self):
        return self.df.drop(columns=self.label_cols).copy()

    @cached_property
    def x_scaled(self):
        # mean centered, unit variance
        x = scale(self.x)
        # reconstruct x as a dataframe
        df = DataFrame(x, columns=self.x.columns.tolist())
        df.index = self.x.index
        return df






if __name__ == "__main__":

    import time

    start = time.perf_counter()

    ds = Dataset()
    df = ds.df
    print(df.shape)
    print(df.head())

    finish = time.perf_counter()

    print((finish - start), "seconds")
