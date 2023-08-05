# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier

from app.classification.pipeline import ClassificationPipeline


class RandomForestPipeline(ClassificationPipeline):

    def __init__(self, ds=None, y_col="is_bot", param_grid=None):
        super().__init__(ds=ds, y_col=y_col, param_grid=param_grid)

        self.model = RandomForestClassifier(random_state=99)
        self.model_dirname = "random_forest"

        self.param_grid = param_grid or {

            # default=100
            "classifier__n_estimators": [50, 100, 150, 250],

            # criterion {"gini", "entropy", "log_loss"}, default="gini"
            # ... The function to measure the quality of a split.
            # ... "gini" for Gini impurity, "log_loss" / "entropy" for Shannon information gain
            "classifier__criterion": ["gini", "log_loss"],

            # min_samples_split (int or float), default=2
            #The minimum number (or percentage) of samples required to split an internal node
            #"classifier__min_samples_split": (2, 10),

            #min_samples_leaf (int or float), default=1
            #... The minimum number of samples required to be at a leaf node.
            # ... A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
            # ... This may have the effect of smoothing the model, especially in regression.
            "classifier__min_samples_leaf": (1,  5,
                                             10, 25,
                                             50, #75,
                                             #90,
                                             100, #110, #125, 150
                                             ),

            # max_depth (int), default=None
            # ... The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            #"classifier__max_depth": (2, 4, 8, 16, None),

            # max_features (int, float) default=None
            # ... The number of features to consider when looking for the best split
            # ... If None, then max_features=n_features.
            #"classifier__max_features": [None, 10, 100],

            # max_leaf_nodes (int), default=None
            # ... Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.
            # ... If None then unlimited number of leaf nodes.
            #"classifier__max_leaf_nodes": [],
        }


if __name__ == "__main__":

    from app.classification import Y_COLS_BINARY
    from app.dataset import Dataset

    ds = Dataset()

    for y_col in Y_COLS_BINARY:

        pipeline = RandomForestPipeline(ds=ds, y_col=y_col)
        pipeline.perform()
