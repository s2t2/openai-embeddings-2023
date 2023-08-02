
import os
from pprint import pprint
from abc import ABC
from functools import cached_property

#import numpy as np
from pandas import Series
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
#import matplotlib.pyplot as plt
import plotly.express as px


from app.dataset import Dataset
from app.classification import CLASSIFICATION_RESULTS_DIRPATH, save_results_json, class_labels
from app.classification.results import ClassificationResults

K_FOLDS = int(os.getenv("K_FOLDS", default="5"))
#X_SCALE = bool(os.getenv("X_SCALE", default="false").lower() == "true")
#SCALER_TYPE = os.getenv("SCALER_TYPE")

FIG_SHOW = bool(os.getenv("FIG_SHOW", default="false").lower() == "true")



class BaseClassifier(ABC):

    def __init__(self, ds=None, x_scale=False, y_col="is_bot", param_grid=None, k_folds=K_FOLDS):

        self.ds = ds or Dataset()
        self.x_scale = x_scale
        self.y_col = y_col

        if self.x_scale:
            self.x = self.ds.x_scaled
        else:
           self.x = self.ds.x

        self.y = self.ds.df[self.y_col]
        self.n_classes = len(set(self.y))

        #if isinstance(y.iloc[0], str):
        #    self.label_binarizer = LabelBinarizer()
        #    y_encoded = self.label_binarizer.fit_transform(y)

        self.k_folds = k_folds

        # values set after training:
        #self.label_binarizer = None # only for categorical / multiclass
        self.class_names = None
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
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, shuffle=True, test_size=0.2, random_state=99)
        print("X TRAIN:", x_train.shape)
        print("Y TRAIN:", y_train.shape)
        print(y_train.value_counts())
        #if self.label_binarizer:
        #    print(Series(self.label_binarizer.inverse_transform(y_train)).value_counts())
        #else:
        #    print(y_train.value_counts())

        pipeline_steps = [("classifier", self.model)]
        #pipeline_steps = []
        #if isinstance(y.iloc[0], str):
        #    # one-hot encoding for categorical labels
        #    label_binarizer = LabelBinarizer()
        #    pipeline_steps.append(("label_binarizer", label_binarizer))
        #    #> TypeError: LabelBinarizer.fit_transform() takes 2 positional arguments but 3 were given
        #pipeline_steps.append(("classifier", self.model))

        pipeline = Pipeline(steps=pipeline_steps)
        scoring ="roc_auc_ovr" if self.n_classes > 2 else  "roc_auc"
        self.gs = GridSearchCV(estimator=pipeline, cv=self.k_folds,
            verbose=10, return_train_score=True, n_jobs=-5, # -1 means using all processors
            scoring=scoring, #"roc_auc",
            param_grid=self.param_grid
        )

        print("-----------------")
        print("TRAINING...")

        self.gs.fit(x_train, y_train)

        print("-----------------")
        print("BEST PARAMS:", self.gs.best_params_)
        print("BEST SCORE:", self.gs.best_score_)
        clf = self.gs.best_estimator_.named_steps["classifier"]
        self.class_names = clf.classes_

        #print("COEFS:")
        #coefs = Series(best_est.coef_[0], index=features).sort_values(ascending=False)

        #breakpoint()

        print("-----------------")
        print("EVALUATION...")

        y_pred = self.gs.predict(x_test)
        #y_pred_proba = self.gs.predict_proba(x_test)

        self.results = ClassificationResults(y_test, y_pred, self.class_names)
        self.results.show_classification_report()

        self.results_json = self.results.as_json
        self.results_json["grid_search"] = {
            "x_scaled": self.x_scale,
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


    def plot_confusion_matrix(self, fig_show=FIG_SHOW, fig_save=True):
        class_names = class_labels(self.y_col, self.class_names)
        cm = self.results.confusion_matrix
        accy = self.results.accy
        f1_macro = self.results.f1_macro
        scaler_title = ", X Scaled" if self.x_scale else ""
        title = f"Confusion Matrix ({self.model_type}{scaler_title})"
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



    #def plot_auc(self, y_test, y_pred):
    #    n_classes = len(set(y_test))
    #    if n_classes == 2:
    #        self.plot_auc_binary(y_test, y_pred)
    #    elif n_classes > 2:
    #        self.plot_auc_multiclass(y_test, y_pred)

    #def plot_auc(self):
    #    if self.n_classes == 2:
    #        self.results.plot_auc_binary()


    #def plot_auc_binary(self, title="Receiver operating characteristic"):
    #    """Plots the ROC characteristic and the AUC Score
    #
    #        See: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    #
    #    """
    #
    #    fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred)
    #    score = auc(fpr, tpr)
    #
    #    fig, ax = plt.subplots(figsize=(10,10))
    #    lw = 2
    #    title = f"ROC curve (area = {round(score, 3)})"
    #    ax.plot(fpr, tpr, color="darkorange", lw=lw, label=title)
    #    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    #    ax.set_xlim([0.0, 1.0])
    #    ax.set_ylim([0.0, 1.0])
    #    plt.xlabel("False Positive Rate")
    #    plt.ylabel("True Positive Rate")
    #    plt.title(title)
    #    plt.legend(loc="lower right")
    #    plt.show()
    #
    #    #if fig_save:
    #    #    plt.savefig()
