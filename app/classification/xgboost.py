

# https://www.nvidia.com/en-us/glossary/data-science/xgboost/
# ... XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library.
# ... It provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems.

# https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost
# ... XGBoost, which is short for "Extreme Gradient Boosting" is a library that provides an efficient implementation of the gradient boosting algorithm.
# ... The main benefit of the XGBoost implementation is computational efficiency and often better model performance.

# https://xgboost.readthedocs.io/en/latest/
# https://xgboost.readthedocs.io/en/latest/python/index.html
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier

from xgboost import XGBClassifier

from app.classification.pipeline import ClassificationPipeline


class XGBoostPipeline(ClassificationPipeline):

    def __init__(self, ds=None, y_col="is_bot", param_grid=None):
        super().__init__(ds=ds, y_col=y_col, param_grid=param_grid)


        #if isinstance(self.y.iloc[0], str):
        #    self.label_binarizer = LabelBinarizer()
        #    self.y = self.label_binarizer.fit_transform(self.y)

        #params = {"random_state": 99}
        #if isinstance(self.y.iloc[0], str):
        #    params["enable_categorical"] = True
        #self.model = XGBClassifier(**params)

        self.model = XGBClassifier(random_state=99)
        self.model_dirname = "xgboost"

        self.param_grid = param_grid or {

            # n_estimators (Optional[int]) – Number of boosting rounds.


            # max_depth (Optional[int]) – Maximum tree depth for base learners.
            #"max_depth": [2, 4, 8, 16],

            # max_leaves – Maximum number of leaves; 0 indicates no limit.
            #"max_leaves": [0, 2, 4, 8, 16]

            # max_bin – If using histogram-based algorithm, maximum number of bins per feature
            #
            # grow_policy – Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow depth-wise. 1: favor splitting at nodes with highest loss change.
            #
            # learning_rate (Optional[float]) – Boosting learning rate (xgb’s “eta”)
            #
            # verbosity (Optional[int]) – The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
            #
            # objective (Union[str, Callable[[numpy.ndarray, numpy.ndarray], Tuple[numpy.ndarray, numpy.ndarray]], NoneType]) – Specify the learning task and the corresponding learning objective or a custom objective function to be used (see note below).
            #
            # booster (Optional[str]) – Specify which booster to use: gbtree, gblinear or dart.
            #
            # tree_method (Optional[str]) – Specify which tree method to use. Default to auto.
            # ... If this parameter is set to default, XGBoost will choose the most conservative option available.
            # ... It’s recommended to study this option from the parameters document tree method
            #
            # n_jobs (Optional[int]) – Number of parallel threads used to run xgboost.
            # ... When used with other Scikit-Learn algorithms like grid search, you may choose which algorithm to parallelize and balance the threads.
            # ... Creating thread contention will significantly slow down both algorithms.
            #
            # gamma (Optional[float]) – (min_split_loss) Minimum loss reduction required to make a further partition on a leaf node of the tree.
            #
            # min_child_weight (Optional[float]) – Minimum sum of instance weight(hessian) needed in a child.
            #
            # max_delta_step (Optional[float]) – Maximum delta step we allow each tree’s weight estimation to be.
            #
            # subsample (Optional[float]) – Subsample ratio of the training instance.
            #
            # sampling_method - Sampling method. Used only by the GPU version of hist tree method.
            #       uniform: select random training instances uniformly.
            #       gradient_based select random training instances with higher probability when the gradient and hessian are larger. (cf. CatBoost)
            #
            # colsample_bytree (Optional[float]) – Subsample ratio of columns when constructing each tree.
            #
            # colsample_bylevel (Optional[float]) – Subsample ratio of columns for each level.
            #
            # colsample_bynode (Optional[float]) – Subsample ratio of columns for each split.
            #
            # reg_alpha (Optional[float]) – L1 regularization term on weights (xgb’s alpha).
            #"reg_alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],

            # reg_lambda (Optional[float]) – L2 regularization term on weights (xgb’s lambda).
            #"reg_lambda": [0.0001, 0.001, 0.01, 0.1, 1, 10],

            # scale_pos_weight (Optional[float]) – Balancing of positive and negative weights.
            #
            # base_score (Optional[float]) – The initial prediction score of all instances, global bias.


            #enable_categorical (bool) –
            #    Experimental support for categorical data. When enabled, cudf/pandas.DataFrame should be used to specify categorical data type. Also, JSON/UBJSON serialization format is required.
            #
            #feature_types (Optional[FeatureTypes]) –
            #    Used for specifying feature types without constructing a dataframe. See DMatrix for details.
            #
            #max_cat_to_onehot (Optional[int]) –
            #    A threshold for deciding whether XGBoost should use one-hot encoding based split for categorical data. When number of categories is lesser than the threshold then one-hot encoding is chosen, otherwise the categories will be partitioned into children nodes. Also, enable_categorical needs to be set to have categorical feature support. See Categorical Data and Parameters for Categorical Feature for details.
            #
            #max_cat_threshold (Optional[int]) –
            #    Maximum number of categories considered for each split. Used only by partition-based splits for preventing over-fitting. Also, enable_categorical needs to be set to have categorical feature support. See Categorical Data and Parameters for Categorical Feature for details.
            #
            #multi_strategy (Optional[str]) –
            #    The strategy used for training multi-target models, including multi-target regression and multi-class classification. See Multiple Outputs for more information.
            #    one_output_per_tree: One model for each target.
            #    multi_output_tree: Use multi-target trees.
            #
            #eval_metric (Optional[Union[str, List[str], Callable]]) –
            #    Metric used for monitoring the training result and early stopping. It can be a string or list of strings as names of predefined metric in XGBoost (See doc/parameter.rst), one of the metrics in sklearn.metrics, or any other user defined metric that looks like sklearn.metrics.
            #    If custom objective is also provided, then custom metric should implement the corresponding reverse link function.
            #    Unlike the scoring parameter commonly used in scikit-learn, when a callable object is provided, it’s assumed to be a cost function and by default XGBoost will minimize the result during early stopping.
            #    For advanced usage on Early stopping like directly choosing to maximize instead of minimize, see xgboost.callback.EarlyStopping.
            #    See Custom Objective and Evaluation Metric for more.
            #
            #early_stopping_rounds (Optional[int]) –
            #    Activates early stopping. Validation metric needs to improve at least once in every early_stopping_rounds round(s) to continue training. Requires at least one item in eval_set in fit().
            #    If early stopping occurs, the model will have two additional attributes: best_score and best_iteration. These are used by the predict() and apply() methods to determine the optimal number of trees during inference. If users want to access the full model (including trees built after early stopping), they can specify the iteration_range in these inference methods. In addition, other utilities like model plotting can also use the entire model.
            #    If you prefer to discard the trees after best_iteration, consider using the callback function xgboost.callback.EarlyStopping.
            #    If there’s more than one item in eval_set, the last entry will be used for early stopping. If there’s more than one metric in eval_metric, the last metric will be used for early stopping.

        }








if __name__ == "__main__":

    from app.classification import Y_COLS_BINARY
    from app.dataset import Dataset

    ds = Dataset()

    for y_col in Y_COLS_BINARY:

        pipeline = XGBoostPipeline(ds=ds, y_col=y_col)
        pipeline.perform()
