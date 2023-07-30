# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn.tree import DecisionTreeClassifier

from app.dataset import Dataset
from app.classification.base import BaseClassifier

class MyRandomForest(BaseClassifier):

    def __init__(self, ds=None, y_col="is_bot", param_grid=None):
        super().__init__(ds=ds, y_col=y_col, param_grid=param_grid)

        self.model = DecisionTreeClassifier(random_state=99)
        self.model_type = "Decision Tree"
        self.model_dirname = "decision_tree"

        self.param_grid = param_grid or {
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
            "classifier__min_samples_leaf": (1, #5,
                                             10, #25,
                                             50, #75,
                                             90,
                                             100, 110, #125, 150
                                             ),

            # max_depth (int), default=None
            # ... The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            #"classifier__max_depth": (1, 2, 3, 4, 5),

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

    ds = Dataset()

    y_cols = [
        "is_bot", #"opinion_community", "is_bom_overall", "is_bom_astroturf",
        #"fourway_label", "bom_overall_fourway_label", "bom_astroturf_fourway_label"
    ]

    for y_col in y_cols:

        clf = MyRandomForest(ds=ds, y_col=y_col)

        clf.train_eval()
