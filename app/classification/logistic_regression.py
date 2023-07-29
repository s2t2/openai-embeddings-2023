
import os
from pprint import pprint

# https://github.com/s2t2/titanic-survival-py/blob/master/app/classifier.py
#from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report, roc_auc_score

from app.classification import CLASSIFICATION_RESULTS_DIRPATH, save_results_json

K_FOLDS = int(os.getenv("K_FOLDS", default="5"))
#X_SCALE = bool(os.getenv("X_SCALE", default="false").lower() == "true")
#SCALER_TYPE = os.getenv("SCALER_TYPE")




from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

def plot_confusion_matrix(clf, y_test, y_pred, y_col, img_filepath=None):
    """Params
        clf : an sklearn classifier (after it has been trained)

        y_col : the column name of y values (for plot labeling purposes)
    """

    classes = clf.classes_
    BOT_LABELS_MAP = {True:"Bot", False:"Human"}
    OPINION_LABELS_MAP = {0:"Anti-Trump", 1:"Pro-Trump"}
    LABELS_MAP = {"is_bot": BOT_LABELS_MAP, "is_bom_overall": BOT_LABELS_MAP, "is_bom_astroturf": BOT_LABELS_MAP, "opinion_community":OPINION_LABELS_MAP}
    if y_col in LABELS_MAP.keys():
        labels_map = LABELS_MAP[y_col]
        class_names = [labels_map[l] for l in classes]
    else:
        class_names = classes

    cm = confusion_matrix(y_test, y_pred)
    #cm = cm.astype(int)
    # Returns: Confusion matrix whose i-th row and j-th column entry
    # ... indicates the number of samples with true label being i-th class and predicted label being j-th class.
    # Interpretation: actual value on rows, predicted value on cols

    #sns.set(rc = {'figure.figsize':(10,10)})
    #sns.heatmap(cm,
    #            square=True,
    #            annot=True, fmt="d",
    #            xticklabels=classes,
    #            yticklabels=classes,
    #            cbar=True, cmap= "Blues" #"Blues" #"viridis_r" #"rocket_r" # r for reverse
    #)
    #plt.ylabel("True Value") # cm rows are true
    #plt.xlabel("Predicted Value") # cm cols are preds
    #plt.title("Confusion Matrix on Test Data (Logistic Regression)")
    #plt.show()
    #if img_filepath:
    #    plt.savefig(img_filepath, format=None)

    #title = f"Confusion Matrix (Logistic Regression, y='{y_col}')"
    title = f"Confusion Matrix (Logistic Regression)"
    #title += f"<br><sup>Scaler: None, Label Col: '{y_col}'</sup>"
    title += f"<br><sup>Y: '{y_col}'</sup>"

    labels = {"x": "Predicted", "y": "Actual"}
    fig = px.imshow(cm, x=class_names, y=class_names, height=450, color_continuous_scale="Blues", labels=labels, text_auto=True)
    fig.update_layout(title={'text': title, 'x':0.485, 'xanchor': 'center'})
    fig.show()

    if img_filepath:
        fig.write_image(img_filepath)
        fig.write_html(img_filepath.replace(".png", ".html"))






if __name__ == "__main__":

    from app.dataset import Dataset

    ds = Dataset()

    #x_scale = False
    #if x_scale:
    #    x = ds.x_scaled
    #else:
    #    x = ds.x
    x = ds.x

    y_cols = [
        "is_bot", "opinion_community", "is_bom_overall", "is_bom_astroturf",
        # "bot_label", #"opinion_label", "bom_overall_label", "bom_astroturf_label",
        "fourway_label", "bom_overall_fourway_label", "bom_astroturf_fourway_label"
    ]
    for y_col in y_cols:
        y = ds.df[y_col]
        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2, random_state=99)
        print("X TRAIN:", x_train.shape)
        print("Y TRAIN:", y_train.shape)
        print(y_train.value_counts())

        clf = LogisticRegression(random_state=99)
        pipeline_steps = [
            # NO SCALER #> ____
            #("scaler", StandardScaler()),
            #("scaler", MinMaxScaler()),
            #("scaler", RobustScaler()),

            ("classifier", clf)
        ]
        pipeline = Pipeline(steps=pipeline_steps)

        param_grid = {
            # default max_iter is 100
            "classifier__max_iter": [5, 15,
                                    20, 25, 30, 35,
                                    50, 100, 250],
            "classifier__solver": ["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
        }
        k_folds=K_FOLDS
        gs = GridSearchCV(estimator=pipeline, cv=k_folds, verbose=10, return_train_score=True, n_jobs=-1, # -1 means using all processors
            scoring="roc_auc",
            param_grid=param_grid
        )

        # TRAINING

        print("-----------------")
        print("TRAINING...")
        gs.fit(x_train, y_train)

        print("-----------------")
        print("BEST PARAMS:", gs.best_params_)
        print("BEST SCORE:", gs.best_score_)

        print("-----------------")
        print("EVALUATION...")

        results = {}
        results["grid_search"] = {
            "sclaler_type": None,
            "classifier_type": clf.__class__.__name__, #> "LogisticRegression"
            "k_folds": k_folds,
            "param_grid": param_grid,
            "best_params": gs.best_params_,
            "best_score": gs.best_score_
        }

        #y_pred = gs.predict_proba(x_test)[:,1]
        y_pred = gs.predict(x_test)

        print(classification_report(y_test, y_pred)) # print in human readable format
        results["classification_eport"] = classification_report(y_test, y_pred, output_dict=True)

        # use with numeric data only, not labels
        if not isinstance(y_test.iloc[0], str):
            results["roc_auc_score"] = roc_auc_score(y_test, y_pred)
            print("ROC AUC SCORE:", results["roc_auc_score"])

        #cm = confusion_matrix(y_test, y_pred)
        #results["confusion_matrix"] = cm

        pprint(results)

        # SAVE RESULTS

        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col)
        os.makedirs(results_dirpath, exist_ok=True)

        json_filepath = os.path.join(results_dirpath, "results.json")
        save_results_json(results, json_filepath)

        # PLOT RESULTS

        img_filepath = os.path.join(results_dirpath, "confusion.png")
        plot_confusion_matrix(clf=gs.best_estimator_, y_test=y_test, y_pred=y_pred, y_col=y_col, img_filepath=img_filepath)
