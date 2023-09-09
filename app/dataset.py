
import os
from functools import cached_property

from pandas import read_csv, DataFrame, isnull
from sklearn.preprocessing import scale #, StandardScaler

from app import DATA_DIRPATH

CSV_FILEPATH = os.path.join(DATA_DIRPATH, "text-embedding-ada-002", "botometer_sample_openai_tweet_embeddings_20230724.csv.gz")

FEATURE_COLS = [str(i) for i in list(range(0, 1536))] # 0 through 1535

DATASET_VERSION = "20230909" # downstream dataset version. bump this when you update the cols


class Dataset():
    def __init__(self, csv_filepath=CSV_FILEPATH, feature_cols=FEATURE_COLS, version=DATASET_VERSION):
        #self.title = "OpenAI Embeddings"
        self.csv_filepath = csv_filepath # upstream (reading in)
        self.feature_cols = feature_cols
        self.version = version # downstream (writing out)

        # consider:
        # self.y_col
        #self.bom_overall_threshold
        #self.bom_astroturf_threshold
        #self.toxicity_threshold
        #self.news_threshold


    @cached_property
    def df(self):
        df = read_csv(self.csv_filepath)

        df.rename(columns={"group_label": "sixway_label"}, inplace=True)

        #
        # BINARY LABELS
        #

        df["is_bom_overall"] = df["bom_overall"].round() # todo: check robustness
        df["is_bom_astroturf"] = df["bom_astroturf"].round() # todo: check robustness

        df["bom_overall_label"] = df["is_bom_overall"].map({1:"Bot", 0:"Human"})
        df["bom_astroturf_label"] = df["is_bom_astroturf"].map({1:"Bot", 0:"Human"})


        toxic_threshold = 0.1 # todo: check robustness
        df["is_toxic"] = df["avg_toxicity"] >= toxic_threshold
        df["is_toxic"] = df["is_toxic"].map({True: 1, False :0 })
        df["toxic_label"] = df["is_toxic"].map({1: "Toxic", 0 :"Non-toxic"})

        # there are null avg_fact_score, so we only apply operation if not null, and leave nulls
        fact_threshold = 3.0 # set threshold and check robustness
        df["is_factual"] = df["avg_fact_score"].apply(lambda score: score if isnull(score) else score >= fact_threshold)
        df["is_factual"] = df["is_factual"].map({True: 1, False :0 })
        df["factual_label"] = df["is_factual"].map({1: "High Quality", 0 :"Low Quality" })
        ## check robustness
        #for fact_threshold in [2.0, 2.5, 3.0]:
        #    threshold_id = str(fact_threshold).replace(".", "") # 2.5 > "25"
        #    # there are null avg_fact_score, so we only apply operation if not null, and leave nulls
        #    colname = f"is_factual_{threshold_id}"
        #    df[colname] = df["avg_fact_score"].apply(lambda score: score if isnull(score) else score >= fact_threshold)
        #    df[colname] = df[colname].map({True: 1, False :0 })
        #    df[f"{colname}_label"] = df["is_factual"].map({1: f"High Quality News", 0 :"Low Quality News"})

        #
        # MULTI-CLASS LABELS
        #

        df["fourway_label"] = df["opinion_label"] + " " + df["bot_label"]
        df["bom_overall_fourway_label"] = df["opinion_label"] + " " + df["bom_overall_label"]
        df["bom_astroturf_fourway_label"] = df["opinion_label"] + " " + df["bom_astroturf_label"]

        df["allway_label"] = df["opinion_label"] + " " + df["bot_label"] + " " + df["toxic_label"] + " " + df["factual_label"]

        return df

    @cached_property
    def labels_df(self):
        label_cols = list(set(self.df.columns.tolist()) - set(self.feature_cols))
        return self.df[label_cols].copy()

    @cached_property
    def x(self):
        return self.df[self.feature_cols].copy()

    @cached_property
    def x_scaled(self):
        # mean centered, unit variance
        x = scale(self.x)
        # reconstruct x as a dataframe
        df = DataFrame(x, columns=self.x.columns.tolist())
        df.index = self.x.index
        return df

    #@cached_property
    #def feature_names(self):
    #    # FYI - PCA get_feature_names_out only works if the feature names are strings
    #    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #    # update: consider returning self.feature_cols
    #    return [str(colname) for colname in self.x.columns.tolist()]





if __name__ == "__main__":

    import time

    start = time.perf_counter()

    ds = Dataset()
    print("DATASET VERSION:", ds.version)

    df = ds.df
    print(df.shape)
    print(df.head())

    finish = time.perf_counter()

    print((finish - start), "seconds")
