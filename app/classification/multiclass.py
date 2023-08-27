# https://scikit-learn.org/stable/modules/multiclass.html
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html


from app.classification.logistic_regression import LogisticRegressionPipeline
from app.classification.random_forest import RandomForestPipeline
from app.classification.xgboost import XGBoostPipeline

if __name__ == "__main__":

    from app.classification import Y_COLS_MULTICLASS
    from app.dataset import Dataset

    ds = Dataset()

    for y_col in Y_COLS_MULTICLASS:

        pipeline = LogisticRegressionPipeline(ds=ds, y_col=y_col)
        pipeline.perform()

        pipeline = XGBoostPipeline(ds=ds, y_col=y_col)
        pipeline.perform()

        pipeline = RandomForestPipeline(ds=ds, y_col=y_col)
        pipeline.perform()
