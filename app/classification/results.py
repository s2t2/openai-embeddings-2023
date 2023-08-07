
from functools import cached_property

from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, RocCurveDisplay


class ClassificationResults:
    """A helper class for reporting on classification results.

        Since metrics like accuracy_score are already included in the classification report, let's get from there instead of re-computing via sklearn metric function.
    """

    def __init__(self, y_test, y_pred, y_pred_proba, class_names):
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba

        self.class_names = class_names #or sorted(list(set(self.y_test)))


    @cached_property
    def classification_report(self):
        return classification_report(self.y_test, self.y_pred, output_dict=True)

    def show_classification_report(self):
         print(classification_report(self.y_test, self.y_pred))

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
        # roc_auc_score uses average="macro" by default
        # ... works with with numeric / boolean class labels, not string categorical class labels
        # ... but  we can one-hot encode categorical class labels to overcome ValueError: could not convert string to float

        #if not isinstance(self.y_test.iloc[0], str):
        #    return roc_auc_score(self.y_test, self.y_pred)
        #else:
#
        #    y_test_encoded = label_binarize(self.y_test, classes=self.class_names)
        #    y_pred_encoded = label_binarize(self.y_pred, classes=self.class_names)
        #    return roc_auc_score(y_test_encoded, y_pred_encoded)

        return roc_auc_score(self.y_test, self.y_pred)

    @cached_property
    def roc_auc_score_proba(self):
        # y_score : Target scores. array-like of shape (n_samples,) or (n_samples, n_classes)
        # In the binary case, it corresponds to an array of shape (n_samples,).
        # ... Both probability estimates and non-thresholded decision values can be provided.
        # ... The probability estimates correspond to the **probability of the class with the greater label**, i.e. estimator.classes_[1] and thus estimator.predict_proba(X, y)[:, 1].
        # ... The decision values corresponds to the output of estimator.decision_function(X, y).
        # ... See more information in the User guide <roc_auc_binary>;
        # In the multiclass case, it corresponds to an array of shape (n_samples, n_classes)
        # ... of probability estimates provided by the predict_proba method.
        # ... The probability estimates **must** sum to 1 across the possible classes.
        # ... In addition, the order of the class scores must correspond to the order of labels, if provided, or else to the numerical or lexicographical order of the labels in y_true.
        # ... See more information in the User guide <roc_auc_multiclass>;

        y_pred_proba_pos = self.y_pred_proba[:,1]
        return roc_auc_score(self.y_test, y_pred_proba_pos)

    @cached_property
    def roc_curve(self):
        y_pred_proba_pos = self.y_pred_proba[:,1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba_pos)
        return fpr, tpr, thresholds

    @cached_property
    def roc_curve_auc(self):
        """This is the same as roc_auc_score_proba"""
        fpr, tpr, _ = self.roc_curve
        return auc(fpr, tpr)



    @cached_property
    def as_json(self):
        return {
            "classification_report": self.classification_report,
            "class_names": self.class_names.tolist(),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "roc_auc_score": self.roc_auc_score,
            "roc_auc_score_proba": self.roc_auc_score_proba,
            "roc_curve_auc": self.roc_curve_auc
        }
