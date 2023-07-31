
from functools import cached_property

from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, RocCurveDisplay


class ClassificationResults:
    """A helper class for reporting on classification results.

        Since metrics like accuracy_score are already included in the classification report, let's get from there instead of re-computing via sklearn metric function.
    """

    def __init__(self, y_test, y_pred, class_names):
        self.y_test = y_test
        self.y_pred = y_pred

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

        if not isinstance(self.y_test.iloc[0], str):
            return roc_auc_score(self.y_test, self.y_pred)
        else:

            y_test_encoded = label_binarize(self.y_test, classes=self.class_names)
            y_pred_encoded = label_binarize(self.y_pred, classes=self.class_names)
            return roc_auc_score(y_test_encoded, y_pred_encoded)

    @cached_property
    def as_json(self):
        return {
            "classification_report": self.classification_report,
            "class_names": self.class_names.tolist(),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "roc_auc_score": self.roc_auc_score
        }
