
from app.classification.logistic_regression import LogisticRegressionPipeline
from app.classification.random_forest import RandomForestClassifier
from app.classification.xgboost import XGBClassifier

if __name__ == "__main__":

    from app.classification import Y_COLS_MULTICLASS
    from app.dataset import Dataset

    ds = Dataset()

    for y_col in Y_COLS_MULTICLASS:

        pipeline = LogisticRegressionPipeline(ds=ds, y_col=y_col)
        pipeline.perform()

        pipeline = RandomForestClassifier(ds=ds, y_col=y_col)
        pipeline.perform()

        pipeline = XGBClassifier(ds=ds, y_col=y_col)
        pipeline.perform()
