
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression

from app.classification.pipeline import ClassificationPipeline

class LogisticRegressionPipeline(ClassificationPipeline):

    def __init__(self, ds=None, y_col="is_bot", param_grid=None, results_dirpath=None):
        super().__init__(ds=ds, y_col=y_col, param_grid=param_grid, results_dirpath=results_dirpath)

        self.model = LogisticRegression(random_state=99)
        self.model_dirname = "logistic_regression"

        self.param_grid = param_grid or {

            #"classifier__penalty": ["l1", "l2", None],

            # C (float), default=1.0
            # Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
            #"classifier__C": [0.5, 1, 2, 10],

            # default max_iter is 100
            "classifier__max_iter": [#5, #15,
                                    #20,
                                    25, #30, #35,
                                    #50,
                                    #100, #250,
                                    #500,
                                    1000
                            ],
            #"classifier__solver": ["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
        }



if __name__ == "__main__":

    from app.classification import Y_COLS_BINARY
    from app.dataset import Dataset

    ds = Dataset()

    for y_col in Y_COLS_BINARY:

        pipeline = LogisticRegressionPipeline(ds=ds, y_col=y_col)
        pipeline.perform()
