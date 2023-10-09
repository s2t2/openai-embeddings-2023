
from functools import cached_property

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, RocCurveDisplay


class ClassificationResults:
    """A helper class for reporting on classification results.

        Since metrics like accuracy_score are already included in the classification report, let's get from there instead of re-computing via sklearn metric function.
    """

    def __init__(self, y_test, y_pred, y_pred_proba, class_names, class_labels=None):
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        self.class_names = class_names #or sorted(list(set(self.y_test)))
        self.is_multiclass = len(self.class_names) >= 3
        self.class_labels = class_labels or class_names #or sorted(list(set(self.y_test)))
        self.class_labels =  [str(l) for l in self.class_labels] # ensure values are strings (for classification report)

    @cached_property
    def classification_report(self):
        return classification_report(self.y_test, self.y_pred, target_names=self.class_labels, output_dict=True)

    def show_classification_report(self):
        print(classification_report(self.y_test, self.y_pred, target_names=self.class_labels))

    @cached_property
    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_pred)

    @cached_property
    def accy(self):
        return round(self.classification_report["accuracy"], 3)

    @cached_property
    def f1_macro(self):
        return round(self.classification_report["macro avg"]["f1-score"], 3)

    @cached_property
    def f1_weighted(self):
        return round(self.classification_report["weighted avg"]["f1-score"], 3)

    @cached_property
    def roc_auc_score(self):
        """NOTE: roc_auc_score uses average='macro' by default"""

        #if isinstance(self.y_test.iloc[0], str):
        #    # roc_auc_score works with with numeric / boolean class labels, not string categorical class labels (will throw ValueError: could not convert string to float)
        #    # ... but we can one-hot encode categorical class labels to overcome the error
        #    y_test_encoded = label_binarize(self.y_test, classes=self.class_names)
        #    y_pred_encoded = label_binarize(self.y_pred, classes=self.class_names)
        #    return roc_auc_score(y_test_encoded, y_pred_encoded)
        #else:
        #    return roc_auc_score(self.y_test, self.y_pred)
        #
        # UPDATE: we are converting string values at the beginning, so this is no longer necessary

        if self.is_multiclass:
           return roc_auc_score(y_true=self.y_test, y_score=self.y_pred_proba, multi_class="ovr")
        else:
            y_pred_proba_pos = self.y_pred_proba[:,1] # positive class (for binary classification)
            return roc_auc_score(y_true=self.y_test, y_score=y_pred_proba_pos)

    #@cached_property
    #def roc_auc_score_proba(self):
    #    # y_score : Target scores. array-like of shape (n_samples,) or (n_samples, n_classes)
    #    # In the binary case, it corresponds to an array of shape (n_samples,).
    #    # ... Both probability estimates and non-thresholded decision values can be provided.
    #    # ... The probability estimates correspond to the **probability of the class with the greater label**, i.e. estimator.classes_[1] and thus estimator.predict_proba(X, y)[:, 1].
    #    # ... The decision values corresponds to the output of estimator.decision_function(X, y).
    #    # ... See more information in the User guide <roc_auc_binary>;
    #    # In the multiclass case, it corresponds to an array of shape (n_samples, n_classes)
    #    # ... of probability estimates provided by the predict_proba method.
    #    # ... The probability estimates **must** sum to 1 across the possible classes.
    #    # ... In addition, the order of the class scores must correspond to the order of labels, if provided, or else to the numerical or lexicographical order of the labels in y_true.
    #    # ... See more information in the User guide <roc_auc_multiclass>;
    #
    #    #y_pred_proba_pos = self.y_pred_proba[:,1] # take the second of two columns, because it wants the scores for the positive class
    #    #return roc_auc_score(self.y_test, y_pred_proba_pos
    #
    #    # ValueError: multi_class must be in ('ovo', 'ovr'). 'raise' is the default
    #    #multi_class = "ovr" if len(self.class_names) >= 3 else "raise"
    #
    #    if self.is_multiclass:
    #        # multiclass, use one-hot encoded, overcomes: numpy.AxisError: axis 1 is out of bounds for array of dimension 1
    #        y_test_encoded = label_binarize(self.y_test, classes=self.class_names)
    #        y_pred_encoded = label_binarize(self.y_pred, classes=self.class_names)
    #        return roc_auc_score(y_test_encoded, y_pred_encoded, multi_class="ovr", average="macro")
    #    else:
    #        y_pred_proba_pos = self.y_pred_proba[:,1] # take the second of two columns, because it wants the scores for the positive class
    #        return roc_auc_score(self.y_test, y_pred_proba_pos)




    @cached_property
    def roc_curve(self):
        """binary classification only"""
        y_pred_proba_pos = self.y_pred_proba[:,1] # second column represents the positive class
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba_pos)
        return fpr, tpr, thresholds

    @cached_property
    def roc_curve_auc(self):
        """This is the same as roc_auc_score_proba (binary classification only)"""
        fpr, tpr, _ = self.roc_curve
        return auc(fpr, tpr)


    @cached_property
    def as_json(self):
        return {
            "class_names": [str(i) for i in self.class_names], # convert numpy int64 (which is not serializable)
            "class_labels": self.class_labels,
            "classification_report": self.classification_report,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "roc_auc_score": self.roc_auc_score,
            #"roc_curve_auc": self.roc_curve_auc,
            #"roc_auc_score_proba": self.roc_auc_score_proba,
        }

    #@cached_property
    #def predictions_df(self) -> DataFrame:
    #    if self.is_multiclass:
    #        breakpoint()
    #        # 'numpy.ndarray' object has no attribute 'index'
    #        #return None # TODO:
    #        df = DataFrame({"y_test": self.y_test, "y_pred": self.y_pred}, index=self.y_test.index)
    #    else:
    #        df = DataFrame({"y_test": self.y_test, "y_pred": self.y_pred}, index=self.y_test.index)
    #        return df
