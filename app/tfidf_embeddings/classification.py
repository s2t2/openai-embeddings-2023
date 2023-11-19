
import os

from app import RESULTS_DIRPATH
from app.classification import Y_COLS, Y_COLS_BINARY, Y_COLS_MULTICLASS
from app.classification.logistic_regression import LogisticRegressionPipeline
from app.classification.random_forest import RandomForestPipeline
from app.classification.xgboost import XGBoostPipeline

from app.tfidf_embeddings.pipeline import TextEmbeddingPipeline

CLASSIFICATION_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "tfidf_classification")


if __name__ == "__main__":


    from app.dataset import Dataset

    ds = Dataset()
    df = ds.df
    df.index = df["user_id"]

    pipeline = TextEmbeddingPipeline(corpus=df["tweet_texts"])
    pipeline.perform()

    # USE TFIDF EMBEDDINGS

    x = pipeline.embbedings_df
    print(x.shape)

    breakpoint()
    tf_ds = Dataset()

    will_upload = True
    for y_col in Y_COLS:
        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, "logistic_regression")
        pipeline = LogisticRegressionPipeline(ds=tf_ds, y_col=y_col, results_dirpath=results_dirpath, will_upload=will_upload)
        pipeline.perform()

        #continue

        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, "xgboost")
        pipeline = XGBoostPipeline(ds=tf_ds, y_col=y_col, results_dirpath=results_dirpath, will_upload=will_upload)
        pipeline.perform()

        # the slowest can go last:
        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, "random_forest")
        pipeline = RandomForestPipeline(ds=tf_ds, y_col=y_col, results_dirpath=results_dirpath, will_upload=will_upload)
        pipeline.perform()
