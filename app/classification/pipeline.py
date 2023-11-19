
import os
from pprint import pprint
from abc import ABC
from functools import cached_property

import numpy as np
from pandas import Series, DataFrame
import plotly.express as px
import plotly.graph_objs as go

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc

from app import save_results_json
from app.colors import ORANGES
from app.dataset import Dataset
from app.model_storage import ModelStorage
from app.classification import CLASSIFICATION_RESULTS_DIRPATH, class_labels
from app.classification.results import ClassificationResults


K_FOLDS = int(os.getenv("K_FOLDS", default="5"))
#X_SCALE = bool(os.getenv("X_SCALE", default="false").lower() == "true")
#SCALER_TYPE = os.getenv("SCALER_TYPE")

FIG_SHOW = bool(os.getenv("FIG_SHOW", default="false").lower() == "true")
FIG_SAVE = bool(os.getenv("FIG_SAVE", default="true").lower() == "true")



class ClassificationPipeline(ABC):
    """Supports binary and multiclass classification."""

    def __init__(self, ds=None, x_scale=False, y_col="is_bot", param_grid=None, k_folds=K_FOLDS, results_dirpath=None, will_upload=True):

        self.ds = ds or Dataset()
        self.x_scale = x_scale
        self.y_col = y_col

        if self.x_scale:
            self.x = self.ds.x_scaled
        else:
           self.x = self.ds.x

        self.y = self.ds.df[self.y_col]
        # if there are null values, consider imputing them or dropping, based on specified strategy
        if self.y.isna().sum() > 0:
            # "drop" strategy
            self.y.dropna(inplace=True) # need to drop x as well
            remaining_indices = self.y.index
            self.x = self.x.loc[remaining_indices]

        self.n_classes = len(set(self.y))
        self.is_multiclass = bool(self.n_classes > 2)

        # if the y labels are strings, let's convert them to numbers (primarily for xgboost)
        if isinstance(self.y.iloc[0], str):
            self.label_encoder = LabelEncoder() # changes strings to numbers starting with 0
            self.y = self.label_encoder.fit_transform(self.y)
            self.class_labels = list(self.label_encoder.classes_) # the original string labels
            self.class_names = list(self.label_encoder.transform(self.class_labels))
        else:
            self.label_encoder = None
            self.class_labels = None
            self.class_names = None

        self.k_folds = k_folds
        self._results_dirpath = results_dirpath

        self.will_upload = bool(will_upload)

        # values set after training:
        self.gs = None
        self.results = None
        self.results_json = {}
        self.storage = None

        # set in child class:
        self.model = None
        self.model_dirname = None
        self.param_grid = param_grid or {}


    @property
    def model_type(self):
        return self.model.__class__.__name__

    def perform(self):
        self.train_eval()
        self.save_results()
        self.save_coefs()
        self.save_predictions()
        self.plot_confusion_matrix()

        if self.is_multiclass:
            self.plot_roc_curve_multiclass()
        else:
            self.plot_roc_curve()

        # upload to cloud storage :-D
        if self.will_upload:
            self.storage = ModelStorage(local_dirpath=self.results_dirpath)
            self.storage.save_and_upload_model(self.model)


    def train_eval(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, shuffle=True, test_size=0.2, random_state=99)
        print("X TRAIN:", self.x_train.shape)
        print("Y TRAIN:", self.y_train.shape)
        #print(self.y_train.value_counts())
        #print(Series(self.y_train).value_counts())

        if self.class_labels:
            print(Series(self.y_train).map(lambda i: self.class_labels[i]).value_counts())
        else:
            print(self.y_train.value_counts())

        steps = [("classifier", self.model)]
        pipeline = Pipeline(steps=steps)
        scoring ="roc_auc_ovr" if self.is_multiclass else "roc_auc"
        self.gs = GridSearchCV(estimator=pipeline, cv=self.k_folds,
            verbose=10, return_train_score=True, n_jobs=-5, # -1 means using all processors
            scoring=scoring, #"roc_auc",
            param_grid=self.param_grid
        )

        print("-----------------")
        print("TRAINING...")

        self.gs.fit(self.x_train, self.y_train)

        print("-----------------")
        print("BEST PARAMS:", self.gs.best_params_)
        print("BEST SCORE:", self.gs.best_score_)

        # overwriting / updating self.model here, because this has the learned coefs, whereas the old self.model does not!
        self.model = self.gs.best_estimator_.named_steps["classifier"]

        self.class_names = self.class_names or list(self.model.classes_)
        self.class_labels = self.class_labels or class_labels(y_col=self.y_col, class_names=self.class_names)

        print("-----------------")
        print("EVALUATION...")

        #breakpoint()

        self.y_pred = self.gs.predict(self.x_test)
        self.y_pred_proba = self.gs.predict_proba(self.x_test)

        self.results = ClassificationResults(self.y_test, self.y_pred, self.y_pred_proba, self.class_names, self.class_labels)
        self.results.show_classification_report()

        self.results_json = {
            "y_col": self.y_col,
            "x_scaled": self.x_scale,
            "grid_search": {
                "model_type": self.model_type,
                "k_folds": self.k_folds,
                "param_grid": self.param_grid,
                "best_params": self.gs.best_params_,
                "best_score": self.gs.best_score_
            },
            "model_params": self.model.get_params() # all params used by the model!
        }
        self.results_json = {**self.results.as_json, **self.results_json} # merge dicts
        pprint(self.results_json)

    @property
    def explainability_json(self) -> dict:
        """implement this in child class"""
        raise NotImplementedError("Please implement in child class. Return a serializable dictionary for JSON conversion.")


    @cached_property
    def predictions_df(self) -> DataFrame:
        """Returns a dataframe of predictions, and corresponding text and labels, for human investigation into the mis-classifications"""
        #if self.is_multiclass:
        if isinstance(self.y_test, np.ndarray):
            # 'numpy.ndarray' object has no attribute 'index'
            index = self.x_test.index # if y test is missing index, use x test index instead
        else:
            index = self.y_test.index

        df = DataFrame({"y_test": self.y_test, "y_pred": self.y_pred}, index=index)
        text_and_labels = self.ds.df[["user_id", "is_bot", "opinion_community", "is_toxic", "is_factual", "fourway_label", "tweet_texts"]]
        df = text_and_labels.merge(df, how="right", left_index=True, right_index=True)
        return df


    def save_results(self):
        json_filepath = os.path.join(self.results_dirpath, "results.json")
        save_results_json(self.results_json, json_filepath)

    def save_predictions(self): # confusion_only=False
        df = self.predictions_df
        csv_filepath = os.path.join(self.results_dirpath, "predictions.csv")
        #if confusion_only:
        #    df = df[df["y_pred"] != df["y_true"]]
        #    csv_filepath = os.path.join(self.results_dirpath, "confusions.csv")
        df.to_csv(csv_filepath, index=False)

    def save_coefs(self):
        json_filepath = os.path.join(self.results_dirpath, "explainability.json")
        save_results_json(self.explainability_json, json_filepath)


    @cached_property
    def results_dirpath(self):
        dirpath = self._results_dirpath or os.path.join(CLASSIFICATION_RESULTS_DIRPATH, self.y_col, self.model_dirname)
        os.makedirs(dirpath, exist_ok=True)
        return dirpath


    def plot_confusion_matrix(self, fig_show=FIG_SHOW, fig_save=FIG_SAVE, showscale=False):
        cm = self.results.confusion_matrix
        accy = self.results.accy
        f1_macro = self.results.f1_macro
        scaler_title = ", X Scaled" if self.x_scale else ""
        title = f"Confusion Matrix ({self.model_type}{scaler_title})"
        title += f"<br><sup>Y: '{self.y_col}' | Accy: {accy} | F-1 Macro: {f1_macro}</sup>"

        fig = px.imshow(cm, x=self.class_labels, y=self.class_labels, height=450,
                        labels={"x": "Predicted", "y": "Actual"},
                        color_continuous_scale="Blues", text_auto=True,
        )
        fig.update_layout(title={'text': title, 'x':0.485, 'xanchor': 'center'})
        fig.update_coloraxes(showscale=showscale) # consider removing the color scale from the image

        if fig_show:
            fig.show()

        if fig_save:
            fig.write_image(os.path.join(self.results_dirpath, "confusion.png"))
            fig.write_html(os.path.join(self.results_dirpath, "confusion.html"))


    def plot_roc_curve(self, fig_show=FIG_SHOW, fig_save=FIG_SAVE, height=500):
        """Plots the ROC characteristic and the AUC Score

            For binary classification.

            See: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

        """

        # roc_curve wants a singe column, but passing it preds leads to only three values coming back
        # ... so we want to use the proba instead (specifically the positive class)
        # proba is two column, with score for each class, so if class labels are [False, True],
        # ... then positive class is in the second column, and we reference with proba[:1]
        # ... https://stackoverflow.com/a/67754984/670433
        #y_pred_proba_pos = self.y_pred_proba[:,1]
        #fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba_pos)
        #score = auc(fpr, tpr)

        fpr, tpr, _ = self.results.roc_curve
        score = self.results.roc_curve_auc

        scaler_title = ", X Scaled" if self.x_scale else ""
        title = f"ROC Curve ({self.model_type}{scaler_title})"
        title += f"<br><sup>Y: '{self.y_col}' | AUC: {score.round(3)}</sup>"

        trace_roc = go.Scatter(x=fpr, y=tpr, mode='lines',
                               line=dict(color='darkorange', width=2),
                               name=f"ROC (AUC = {score.round(3)})"
        )
        trace_diag = go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                line=dict(color='navy', width=2, dash='dash'),
                                name="Chance level (AUC = 0.5)"
        )

        # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Layout.html
        layout = go.Layout(
            title=title, title_x=0.5, # centered
            width=height, height=height, # square
            xaxis=dict(title="False Positive Rate"),
            yaxis=dict(title="True Positive Rate", ticks="inside"),
            showlegend=True,
            legend=dict(x=.98, y=0.02, xanchor='right', yanchor='bottom', bordercolor='gray', borderwidth=1),
        )

        fig = go.Figure(data=[trace_roc, trace_diag], layout=layout)

        if fig_show:
            fig.show()

        if fig_save:
            fig.write_image(os.path.join(self.results_dirpath, "roc_curve.png"))
            fig.write_html(os.path.join(self.results_dirpath, "roc_curve.html"))

        return fig


    def plot_roc_curve_multiclass(self, fig_show=FIG_SHOW, fig_save=FIG_SAVE, height=500):

        # CHART DATA

        label_binarizer = LabelBinarizer().fit(self.y_train)
        y_test_encoded = label_binarizer.transform(self.y_test)
        class_names = self.class_labels #label_binarizer.classes_ # self.class_names

        chart_data = []
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_test_encoded[:,i], self.y_pred_proba[:,i])
            score = auc(fpr, tpr)
            trace = go.Scatter(x=fpr, y=tpr,
                mode='lines',
                line=dict(color=ORANGES[i+2], width=2),
                name=f"'{str(class_name).title()}' vs Rest (AUC = {score.round(3)})"
            )
            chart_data.append(trace)

        trace_diag = go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
            line=dict(color='navy', width=2, dash='dash'),
            name="Chance level (AUC = 0.5)"
        )
        chart_data.append(trace_diag)

        # LAYOUT

        #macro_ovr_score = roc_auc_score(self.y_test, self.y_pred_proba, multi_class="ovr", average="macro")
        macro_ovr_score = self.results.roc_auc_score

        title = f"ROC Curve ({self.model_type})"
        title += f"<br><sup><sup>Y: '{self.y_col}' | AUC (Macro averaged One vs Rest): {round(macro_ovr_score, 3)}</sup>"

        layout = go.Layout(title=title, title_x=0.5, # centered
            width=height, height=height, # square
            xaxis=dict(title="False Positive Rate"),
            yaxis=dict(title="True Positive Rate", ticks="inside"),
            showlegend=True,
            legend=dict(x=.98, y=0.02, xanchor='right', yanchor='bottom', bordercolor='gray', borderwidth=1),
        ) # https://plotly.com/python-api-reference/generated/plotly.graph_objects.Layout.html

        # FIGURE

        fig = go.Figure(data=chart_data, layout=layout)

        if fig_show:
            fig.show()

        if fig_save:
            fig.write_image(os.path.join(self.results_dirpath, "roc_curve.png"))
            fig.write_html(os.path.join(self.results_dirpath, "roc_curve.html"))

        return fig
