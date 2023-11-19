import os

from app import RESULTS_DIRPATH
from app.classification import Y_COLS, Y_COLS_BINARY, Y_COLS_MULTICLASS
from app.classification.logistic_regression import LogisticRegressionPipeline
from app.classification.random_forest import RandomForestPipeline
from app.classification.xgboost import XGBoostPipeline

from app.word2vec_embeddings.pipeline import WORD2VEC_RESULTS_DIRPATH
from app.word2vec_classification.dataset import Word2VecDataset


CLASSIFICATION_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "word2vec_classification")


if __name__ == "__main__":

    ds = Word2VecDataset()

    will_upload = True
    for y_col in Y_COLS:
        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, "logistic_regression")
        pipeline = LogisticRegressionPipeline(ds=ds, y_col=y_col, results_dirpath=results_dirpath, will_upload=will_upload)
        pipeline.perform()

        #continue

        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, "xgboost")
        pipeline = XGBoostPipeline(ds=ds, y_col=y_col, results_dirpath=results_dirpath, will_upload=will_upload)
        pipeline.perform()

        # the slowest can go last:
        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, "random_forest")
        pipeline = RandomForestPipeline(ds=ds, y_col=y_col, results_dirpath=results_dirpath, will_upload=will_upload)
        pipeline.perform()
