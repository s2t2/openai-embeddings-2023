
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


def plot_confusion_matrix(clf, y_test, y_pred, img_filepath=None):
    """Params clf : an sklearn classifier (after it has been trained) """

    classes = clf.classes_

    cm = confusion_matrix(y_test, y_pred)
    # Confusion matrix whose i-th row and j-th column entry indicates the number of samples with
    # ... true label being i-th class and predicted label being j-th class.

    # df = DataFrame(cm, columns=classes, index=classes)

    #sns.set(rc = {'figure.figsize':(10,10)})

    sns.heatmap(cm,
                square=True,
                annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes,
                cbar=True, cmap= "Blues" #"Blues" #"viridis_r" #"rocket_r" # r for reverse
    )

    plt.ylabel("True Value") # cm rows are true
    plt.xlabel("Predicted Value") # cm cols are preds
    plt.title("Confusion Matrix on Test Data (Logistic Regression)")
    plt.show()

    if img_filepath:
        plt.savefig(img_filepath, format=None)







if __name__ == "__main__":

    from app.dataset import Dataset

    ds = Dataset()

    x_scale = False
    if x_scale:
        x = ds.x_scaled
    else:
        x = ds.x

    y_cols = ["is_bot",
              #"opinion_community", "is_bom_overall", "is_bom_astroturf",
              #"fourway_label", "bom_overall_fourway_label", "bom_astroturf_fourway_label"
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

        print(classification_report(y_test, y_pred))
        results["classification_eport"] = classification_report(y_test, y_pred, output_dict=True)

        #Only used for multiclass targets. Determines the type of configuration to use. The default value raises an error, so either 'ovr' or 'ovo' must be passed explicitly.
        #
        #'ovr':
        #Stands for One-vs-rest. Computes the AUC of each class against the rest [3] [4]. This treats the multiclass case in the same way as the multilabel case. Sensitive to class imbalance even when average == 'macro', because class imbalance affects the composition of each of the ‘rest’ groupings.
        #
        #'ovo':
        #Stands for One-vs-one. Computes the average AUC of all possible pairwise combinations of classes [5]. Insensitive to class imbalance when average == 'macro'.
        results["roc_auc_score"] = roc_auc_score(y_test, y_pred)

        print("ROC AUC SCORE:", results["roc_auc_score"])
        #pprint(results)

        # SAVE RESULTS

        results_dirpath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col)
        os.makedirs(results_dirpath, exist_ok=True)

        json_filepath = os.path.join(results_dirpath, "results.json")
        save_results_json(results, json_filepath)

        # PLOT RESULTS

        img_filepath = os.path.join(results_dirpath, "confusion.png")
        breakpoint()
        plot_confusion_matrix(gs.best_estimator_, y_test, y_pred, img_filepath)
