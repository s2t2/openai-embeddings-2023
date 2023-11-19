
import os
from functools import cached_property

from pandas import read_csv

from app import RESULTS_DIRPATH
from app.classification import Y_COLS, Y_COLS_BINARY, Y_COLS_MULTICLASS
from app.classification.logistic_regression import LogisticRegressionPipeline
from app.classification.random_forest import RandomForestPipeline
from app.classification.xgboost import XGBoostPipeline

from app.tfidf_embeddings.pipeline import TextEmbeddingPipeline

CLASSIFICATION_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "tfidf_classification")

class TextDataset():
    """The original dataset interface assumes a CSV file and that's too opinionated"""

    def __init__(self, df, x):
        #self.csv_filepath = None
        #self.label_cols = None
        #self.labels_df = None

        self.df = df
        self.x = x



if __name__ == "__main__":


    from app.dataset import Dataset

    ds = Dataset()
    df = ds.df
    df.index = df["user_id"]

    pipeline = TextEmbeddingPipeline(corpus=df["tweet_texts"])
    pipeline.perform()

    # USE TFIDF EMBEDDINGS

    x = pipeline.embeddings_df
    print(x.shape)

    # dataset api on the fly:
    text_ds = TextDataset(df=df, x=x)

    will_upload = False
    for y_col in Y_COLS:
        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, "logistic_regression")
        pipeline = LogisticRegressionPipeline(ds=text_ds, y_col=y_col, results_dirpath=results_dirpath, will_upload=will_upload, param_grid={

            # C (float), default=1.0
            # Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
            "classifier__C": [
                1, #2, 5,
                10, #25, 50,
                #100
            ],

            # default max_iter is 100
            "classifier__max_iter": [#10, 25,
                                     50,
                                     100,
                                     #250,
                                     500,
                                     #1_000, #5_000, 10_000
                                     ],
        })
        pipeline.perform()

        continue

        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, "xgboost")
        pipeline = XGBoostPipeline(ds=text_ds, y_col=y_col, results_dirpath=results_dirpath, will_upload=will_upload, param_grid={
            'classifier__n_estimators': [50,
                                         100, 150,
                                         250]
        })
        pipeline.perform()

        # the slowest can go last:
        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, "random_forest")
        pipeline = RandomForestPipeline(ds=text_ds, y_col=y_col, results_dirpath=results_dirpath, will_upload=will_upload, param_grid={
            'classifier__n_estimators': [50,
                                         100, 150,
                                         250]
        })
        pipeline.perform()
