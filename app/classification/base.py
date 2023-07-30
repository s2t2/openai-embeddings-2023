
import os
from pprint import pprint
from abc import ABC
from functools import cached_property

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import plotly.express as px

from app.dataset import Dataset
from app.classification import CLASSIFICATION_RESULTS_DIRPATH, save_results_json, CLASSES_MAP
#from app.classification.metrics import plot_confusion_matrix

K_FOLDS = int(os.getenv("K_FOLDS", default="5"))
#X_SCALE = bool(os.getenv("X_SCALE", default="false").lower() == "true")
#SCALER_TYPE = os.getenv("SCALER_TYPE")



class ClassificationResults:

    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred


    @cached_property
    def classification_report(self):
        #if as_json:
        #    print(classification_report(self.y_test, self.y_pred))
        #else:
        #    return classification_report(self.y_test, self.y_pred, output_dict=True)
        #return classification_report(self.y_test, self.y_pred, output_dict=output_dict)
        return classification_report(self.y_test, self.y_pred, output_dict=True)

    def show_classification_report(self):
         print(classification_report(self.y_test, self.y_pred))

    @cached_property
    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_pred)

    def metrics(self):
        # WE HAVE ACCURACY FROM THE CLASSIFICATION REPORT
        #self.results_["metrics"]["accuracy_score"] = accuracy_score(y_test, y_pred)

        # binary vs multiclass classification
        #metric_params = dict(y_true=y_test, y_pred=y_pred)
        #if len(set(y_pred)):
        #    # ValueError: Target is multiclass but average='binary'.
        #    # Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
        #    metric_params["average"] = "..."
        #self.results_["metrics"]["precision_score"] = precision_score(y_test, y_pred)
        #self.results_["metrics"]["recall_score"] = recall_score(y_test, y_pred)
        #self.results_["metrics"]["f1_score"] = f1_score(y_test, y_pred)
        ## YEAH BUT WE WANT TO REPORT ON BOTH THE MACRO AVERAGE AND WEIGHTED AVERAGE
        ## AND THIS INFO IS ALREADY IN THE CLASSIFICATION REPORT,
        ## SO LET'S TAKE IT FROM THERE INSTEAD
        pass

    @cached_property
    def accy(self):
        return round(self.results.classification_report["accuracy"], 3)

    @cached_property
    def f1_macro(self):
        return round(self.results.classification_report["macro avg"]["f1-score"], 3)

    def roc_auc_score(self, average="macro"):
        # use with numeric / boolean data only, not string categorical class labels
        if not isinstance(self.y_test.iloc[0], str):
            return roc_auc_score(self.y_test, self.y_pred)
        else:
            breakpoint()
            # one-hot encoded y labels
            y_test_encoded = label_binarize(self.y_test) # classes=self.class_names
            y_pred_encoded = label_binarize(self.y_pred) # classes=self.class_names
            return roc_auc_score(y_test_encoded, y_pred_encoded, average=average)

    @property
    def as_json(self):
        return {
            "classification_report": self.classification_report,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "roc_auc_score": self.roc_auc_score()
        }



class BaseClassifier(ABC):

    def __init__(self, ds=None, x_scale=False, y_col="is_bot", param_grid=None, k_folds=K_FOLDS):

        self.ds = ds or Dataset()
        self.x_scale = x_scale
        self.y_col = y_col

        self.k_folds = k_folds

        # values set after training:
        self.gs = None
        self.results = None
        self.results_json = {}

        # set in child class:
        self.model = None
        self.model_dirname = None
        self.param_grid = param_grid or {}

    @property
    def model_type(self):
        return self.model.__class__.__name__

    def train_eval(self):
        if self.x_scale:
            x = self.ds.x_scaled
        else:
            x = self.ds.x

        y = self.ds.df[self.y_col]

        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=99)
        print("X TRAIN:", x_train.shape)
        print("Y TRAIN:", y_train.shape)
        print(y_train.value_counts())

        pipeline_steps = [("classifier", self.model)]
        pipeline = Pipeline(steps=pipeline_steps)
        self.gs = GridSearchCV(estimator=pipeline, cv=self.k_folds,
            verbose=10, return_train_score=True, n_jobs=-5, # -1 means using all processors
            scoring="roc_auc",
            param_grid=self.param_grid
        )

        self.gs.fit(x_train, y_train)

        print("-----------------")
        print("BEST PARAMS:", self.gs.best_params_)
        print("BEST SCORE:", self.gs.best_score_)

        print("-----------------")
        print("EVALUATION...")

        y_pred = self.gs.predict(x_test)

        self.results = ClassificationResults(y_test, y_pred) # class_names=self.class_names
        self.results.show_classification_report()

        self.results_json = self.results.as_json
        self.results_json["grid_search"] = {
            "sclaler_type": None,
            "model_type": self.model_type, #self.gs.best_estimator_.named_steps["classifier"].__class__.__name__,
            "k_folds": self.k_folds,
            "param_grid": self.param_grid,
            "best_params": self.gs.best_params_,
            "best_score": self.gs.best_score_
        }
        pprint(self.results_json)


    def save_results(self):
        json_filepath = os.path.join(self.results_dirpath, "results.json")
        save_results_json(self.results_json, json_filepath)


    @cached_property
    def results_dirpath(self):
        dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, self.y_col, self.model_dirname)
        os.makedirs(dirpath, exist_ok=True)
        return dirpath


    def plot_confusion_matrix(self, fig_show=True, fig_save=True):
        """Params

            cm : an sklearn confusion matrix result
                ... Confusion matrix whose i-th row and j-th column entry
                ... indicates the number of samples with true label being i-th class and predicted label being j-th class.
                ... Interpretation: actual value on rows, predicted value on cols

            clf : an sklearn classifier (after it has been trained)

            y_col : the column name of y values (for plot labeling purposes)

            image_filestem : the directory path and file name (excluding file extension)
        """

        clf = self.gs.best_estimator_.named_steps["classifier"]
        class_names = clf.classes_
        # apply custom labels for binary values, to make chart easier to read
        if self.y_col in CLASSES_MAP.keys():
            classes_map = CLASSES_MAP[self.y_col]
            class_names = [classes_map[val] for val in class_names]

        cm = self.results.confusion_matrix
        accy = round(self.results.classification_report["accuracy"], 3)
        f1_macro = round(self.results.classification_report["macro avg"]["f1-score"], 3)
        #f1_weighted = round(self.results.classification_report["weighted avg"]["f1-score"], 3)

        title = f"Confusion Matrix ({self.model_type})"
        title += f"<br><sup>Y: '{self.y_col}' | Accy: {accy} | F-1 Macro: {f1_macro}</sup>"

        fig = px.imshow(cm, x=class_names, y=class_names, height=450,
                        labels={"x": "Predicted", "y": "Actual"},
                        color_continuous_scale="Blues", text_auto=True,
        )
        fig.update_layout(title={'text': title, 'x':0.485, 'xanchor': 'center'})

        if fig_show:
            fig.show()

        if fig_save:
            fig.write_image(os.path.join(self.results_dirpath, "confusion.png"))
            fig.write_html(os.path.join(self.results_dirpath, "confusion.html"))
