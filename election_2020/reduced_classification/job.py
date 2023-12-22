import os

from app import RESULTS_DIRPATH
from app.reduced_dataset import ReducedDataset, REDUCTIONS
from app.classification import Y_COLS, Y_COLS_BINARY, Y_COLS_MULTICLASS
from app.classification.logistic_regression import LogisticRegressionPipeline
from app.classification.random_forest import RandomForestPipeline
from app.classification.xgboost import XGBoostPipeline


if __name__ == "__main__":

    for reducer_name, n_components in REDUCTIONS:
        results_dirname = f"{reducer_name}_{n_components}"
        reducer_type = {"pca": "PCA", "tsne": "T-SNE", "umap":"UMAP"}[reducer_name]

        # not the most ideal that we are loading the dataset multiple times instead of just choosing different col names from a dataset that has been loaded once, however integration with previous dataset API has led us here
        ds = ReducedDataset(reducer_name=reducer_name, n_components=n_components)

        for y_col in Y_COLS:
            results_dirpath = os.path.join(RESULTS_DIRPATH, "reduced_classification", y_col, results_dirname, "logistic_regression")
            pipeline = LogisticRegressionPipeline(ds=ds, y_col=y_col, results_dirpath=results_dirpath)
            pipeline.perform()

            results_dirpath = os.path.join(RESULTS_DIRPATH, "reduced_classification", y_col, results_dirname, "xgboost")
            pipeline = XGBoostPipeline(ds=ds, y_col=y_col, results_dirpath=results_dirpath)
            pipeline.perform()

            # the slowest can go last:
            results_dirpath = os.path.join(RESULTS_DIRPATH, "reduced_classification", y_col, results_dirname, "random_forest")
            pipeline = RandomForestPipeline(ds=ds, y_col=y_col, results_dirpath=results_dirpath)
            pipeline.perform()
