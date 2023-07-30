
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression

from app.dataset import Dataset
from app.classification import Y_COLS
from app.classification.base import BaseClassifier

class MyLogisticRegression(BaseClassifier):

    def __init__(self, ds=None, y_col="is_bot", param_grid=None):
        super().__init__(ds=ds, y_col=y_col, param_grid=param_grid)

        self.model = LogisticRegression(random_state=99)
        self.model_dirname = "logistic_regression"

        self.param_grid = param_grid or {
            # default max_iter is 100
            "classifier__max_iter": [#5, 15,
                                    20,
                                    25, 30, #35,
                                    50, 100, #250, 500
                                    ],
            "classifier__solver": ["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
        }



if __name__ == "__main__":

    ds = Dataset()

    for y_col in Y_COLS:

        clf = MyLogisticRegression(ds=ds, y_col=y_col)

        clf.train_eval()
        clf.save_results()
        clf.plot_confusion_matrix()
