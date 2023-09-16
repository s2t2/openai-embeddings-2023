import os

from app.dataset import Dataset
from app.classification import CLASSIFICATION_RESULTS_DIRPATH, Y_COLS, Y_COLS_BINARY, Y_COLS_MULTICLASS
from app.classification.logistic_regression import LogisticRegressionPipeline
from app.classification.random_forest import RandomForestPipeline
from app.classification.xgboost import XGBoostPipeline



if __name__ == "__main__":

    ds = Dataset()

    for y_col in Y_COLS:
        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, "logistic_regression")
        pipeline = LogisticRegressionPipeline(ds=ds, y_col=y_col, results_dirpath=results_dirpath)
        pipeline.perform()

        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, "xgboost")
        pipeline = XGBoostPipeline(ds=ds, y_col=y_col, results_dirpath=results_dirpath)
        pipeline.perform()

        # the slowest can go last:
        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, "random_forest")
        pipeline = RandomForestPipeline(ds=ds, y_col=y_col, results_dirpath=results_dirpath)
        pipeline.perform()
