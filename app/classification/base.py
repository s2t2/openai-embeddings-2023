
import os
from pprint import pprint
from abc import ABC

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import plotly.express as px

from app.dataset import Dataset
from app.classification import CLASSIFICATION_RESULTS_DIRPATH, save_results_json, CLASSES_MAP
#from app.classification.metrics import plot_confusion_matrix

K_FOLDS = int(os.getenv("K_FOLDS", default="5"))
#X_SCALE = bool(os.getenv("X_SCALE", default="false").lower() == "true")
#SCALER_TYPE = os.getenv("SCALER_TYPE")


class BaseClassifier(ABC):

    def __init__(self, ds=None, x_scale=False, y_col="is_bot", param_grid=None, k_folds=K_FOLDS):

        self.ds = ds or Dataset()
        self.x_scale = x_scale
        self.y_col = y_col

        self.k_folds = k_folds
        self.gs = None
        self.results_ = {}

        # set in child class:
        self.model = None
        self.model_type = None
        self.model_dirname = None
        self.param_grid = param_grid or {}






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

        # MODEL SELECTION

        pipeline_steps = [("classifier", self.model)]
        pipeline = Pipeline(steps=pipeline_steps)
        self.gs = GridSearchCV(estimator=pipeline, cv=self.k_folds, verbose=10, return_train_score=True, n_jobs=-5, # -1 means using all processors
            scoring="roc_auc",
            param_grid=self.param_grid
        )

        # MODEL TRAINING

        print("-----------------")
        print("TRAINING...")
        self.gs.fit(x_train, y_train)

        print("-----------------")
        print("BEST PARAMS:", self.gs.best_params_)
        print("BEST SCORE:", self.gs.best_score_)

        # MODEL EVAL

        print("-----------------")
        print("EVALUATION...")

        model_type = self.model_type or self.gs.best_estimator_.named_steps["classifier"].__class__.__name__

        self.results_["grid_search"] = {
            "sclaler_type": None,
            "model_type": model_type,
            "k_folds": self.k_folds,
            "param_grid": self.param_grid,
            "best_params": self.gs.best_params_,
            "best_score": self.gs.best_score_
        }

        y_pred = self.gs.predict(x_test)

        self.results_["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred)) # print in human readable format

        self.results_["metrics"]["accuracy_score"] = accuracy_score(y_test, y_pred)
        self.results_["metrics"]["precision_score"] = precision_score(y_test, y_pred)
        self.results_["metrics"]["recall_score"] = recall_score(y_test, y_pred)
        self.results_["metrics"]["f1_score"] = f1_score(y_test, y_pred)

        # use with numeric data only, not labels
        if not isinstance(y_test.iloc[0], str):
            roc_auc = roc_auc_score(y_test, y_pred)
            self.results_["metrics"]["roc_auc_score"] = roc_auc
            print("ROC AUC SCORE:", roc_auc)

        cm = confusion_matrix(y_test, y_pred)
        self.results_["confusion_matrix"] = cm.tolist()

        pprint(self.results_)

        # SAVE RESULTS

        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, self.y_col, self.model_dirname)
        os.makedirs(results_dirpath, exist_ok=True)

        json_filepath = os.path.join(results_dirpath, "results.json")
        save_results_json(self.results_, json_filepath)

        # PLOT RESULTS

        img_filestem = os.path.join(results_dirpath, "confusion")
        self.plot_confusion_matrix(cm=cm, img_filestem=img_filestem)



    def plot_confusion_matrix(self, cm, img_filestem=None):
        """Params

            cm : an sklearn confusion matrix result
                ... Confusion matrix whose i-th row and j-th column entry
                ... indicates the number of samples with true label being i-th class and predicted label being j-th class.
                ... Interpretation: actual value on rows, predicted value on cols


            clf : an sklearn classifier (after it has been trained)

            y_col : the column name of y values (for plot labeling purposes)

            image_filepath : ends with ".png"
        """

        clf = self.gs.best_estimator_.named_steps["classifier"]
        classes = clf.classes_
        if self.y_col in CLASSES_MAP.keys():
            classes_map = CLASSES_MAP[self.y_col]
            class_names = [classes_map[val] for val in classes]
        else:
            class_names = classes

        accy = round(self.results_["accuracy_score"], 3)
        f1 = round(self.results_["f1_score"], 3)
        title = f"Confusion Matrix ({clf.__class__.__name__})"
        title += f"<br><sup>Y: '{self.y_col}' | Accy: {accy} | F1: {f1}</sup>"

        labels = {"x": "Predicted", "y": "Actual"}
        fig = px.imshow(cm, x=class_names, y=class_names, height=450, color_continuous_scale="Blues", labels=labels, text_auto=True)
        fig.update_layout(title={'text': title, 'x':0.485, 'xanchor': 'center'})
        fig.show()

        if img_filestem:
            fig.write_image(f"{img_filestem}.png")
            fig.write_html(f"{img_filestem}.html")
