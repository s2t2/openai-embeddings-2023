
import os
from functools import cached_property

from pandas import read_csv, DataFrame, isnull
from sklearn.preprocessing import scale #, StandardScaler

from app import DATA_DIRPATH

CSV_FILEPATH = os.path.join(DATA_DIRPATH, "text-embedding-ada-002", "botometer_sample_openai_tweet_embeddings_20230724.csv.gz")

LABEL_COLS = [
    # in original dataset
    'user_id', 'created_on', 'screen_name_count', 'screen_names', 'status_count', 'rt_count', 'rt_pct',
    'avg_toxicity', 'avg_fact_score', 'opinion_community', 'is_bot', 'is_q',
    'tweet_texts',
    'bom_cap', 'bom_astroturf', 'bom_fake_follower', 'bom_financial', 'bom_other', 'bom_overall', 'bom_self_declared','bom_spammer',

    # virtual attributes
    "is_bom_overall", 'is_bom_astroturf',
    "is_toxic", "is_factual", "is_factual",

    'opinion_label', 'bot_label', 'q_label',
    'bom_overall_label', 'bom_astroturf_label', #'group_label'
    'fourway_label', 'sixway_label', "bom_overall_fourway_label", "bom_astroturf_fourway_label"
]


class Dataset():
    def __init__(self, csv_filepath=CSV_FILEPATH, label_cols=LABEL_COLS):
        #self.title = "OpenAI Embeddings"
        self.csv_filepath = csv_filepath
        self.label_cols = label_cols

    @cached_property
    def df(self):
        df = read_csv(self.csv_filepath)

        df.rename(columns={"group_label": "sixway_label"}, inplace=True)
        #print(df["sixway_label"].value_counts())

        df["fourway_label"] = df["opinion_label"] + " " + df["bot_label"]
        #print(df["fourway_label"].value_counts())

        df["is_bom_overall"] = df["bom_overall"].round()
        df["is_bom_astroturf"] = df["bom_astroturf"].round()
        df["bom_overall_label"] = df["is_bom_overall"].map({1:"Bot", 0:"Human"})
        df["bom_astroturf_label"] = df["is_bom_astroturf"].map({1:"Bot", 0:"Human"})
        df["bom_overall_fourway_label"] = df["opinion_label"] + " " + df["bom_overall_label"]
        df["bom_astroturf_fourway_label"] = df["opinion_label"] + " " + df["bom_astroturf_label"]

        toxic_threshold = 0.1 # set threshold and check robustness
        df["is_toxic"] = df["avg_toxicity"] >= toxic_threshold
        df["is_toxic"] = df["is_toxic"].map({True: 1, False :0 })

        # there are null avg_fact_score, so we only apply operation if not null, and leave nulls
        fact_threshold = 3.0 # set threshold and check robustness
        df["is_factual"] = df["avg_fact_score"].apply(lambda score: score if isnull(score) else score >= fact_threshold)
        df["is_factual"] = df["is_factual"].map({True: 1, False :0 })

        return df

    @cached_property
    def labels(self):
        return self.df[self.label_cols].copy()

    @cached_property
    def labels_slim(self):
        return self.df[["user_id", "is_bot", "opinion_community", "is_q",
                        "bot_label", "opinion_label", "q_label", "fourway_label", "sixway_label"]].copy()

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

    @cached_property
    def feature_names(self):
        # FYI - PCA get_feature_names_out only works if the feature names are strings
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        return [str(colname) for colname in self.x.columns.tolist()]





if __name__ == "__main__":

    import time

    start = time.perf_counter()

    ds = Dataset()
    df = ds.df
    print(df.shape)
    print(df.head())

    finish = time.perf_counter()

    print((finish - start), "seconds")
