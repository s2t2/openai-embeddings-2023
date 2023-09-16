
import os
import json

from pandas import DataFrame

from app import RESULTS_DIRPATH
from app.classification import CLASSIFICATION_RESULTS_DIRPATH, Y_COLS
from app.reduced_dataset import REDUCTIONS
#from app.reduced_classification import

def load_json(json_filepath):
    with open(json_filepath, "r") as json_file:
        return json.load(json_file)


class Results:
    def __init__(self, json_filepath):
        self.json_filepath = json_filepath
        self.data = load_json(self.json_filepath)

    @property
    def model_type(self):
        return self.data["grid_search"]["model_type"]

    @property
    def best_params(self):
        return self.data["grid_search"]["best_params"]

    @property
    def accy(self):
        return round(self.data["classification_report"]["accuracy"], 3)

    @property
    def f1_macro(self):
        return round(self.data["classification_report"]["macro avg"]["f1-score"], 3)

    @property
    def f1_weighted(self):
        return round(self.data["classification_report"]["weighted avg"]["f1-score"], 3)

    @property
    def roc_auc_score(self):
        # return round(self.data["roc_auc_score"], 3)

        if "roc_auc_score_proba" in self.data.keys():
            # deprecated legacy name from old runs. need to re-generate all results again and then update this method to look at roc_auc_score only
            return round(self.data["roc_auc_score_proba"], 3)
        else:
            return round(self.data["roc_auc_score"], 3)



if __name__ == "__main__":

    records = []

    for y_col in Y_COLS:
        for model_name in ["logistic_regression", "random_forest", "xgboost"]:

            #
            # CLASSIFICATION RESULTS
            #

            results_filepath = os.path.join(CLASSIFICATION_RESULTS_DIRPATH, y_col, model_name, "results.json")

            try:
                results = Results(results_filepath)
                record = {
                    # methods:
                    "dataset": "openai_embeddings",
                    "reducer_type": None, #"reducer_name": None,
                    "n_components": 1536,
                    "y_col": y_col,
                    "model_type": results.model_type, #"model_name": model_name,
                    "best_params": results.best_params, # FYI: this is a dict
                    # metrics:
                    "accuracy": results.accy,
                    "f1_macro": results.f1_macro,
                    "f1_weighted": results.f1_weighted,
                    "roc_auc_score": results.roc_auc_score,
                }
                records.append(record)
            except FileNotFoundError as err:
                    print("MISSING:", results_filepath.replace(RESULTS_DIRPATH, ""))

            #
            # REDUCED CLASSIFICATION RESULTS
            #

            for reducer_name, n_components in REDUCTIONS:
                results_dirname = f"{reducer_name}_{n_components}"
                reducer_type = {"pca": "PCA", "tsne": "T-SNE", "umap":"UMAP"}[reducer_name]

                results_filepath = os.path.join(RESULTS_DIRPATH, "reduced_classification", y_col,
                                                results_dirname, model_name, "results.json")
                try:
                    results = Results(results_filepath)
                    record = {
                        # methods:
                        "dataset": results_dirname,
                        "reducer_type": reducer_type, #"reducer_name": reducer_name,
                        "n_components": n_components,
                        "y_col": y_col,
                        "model_type": results.model_type, # "model_name": model_name,
                        "best_params": results.best_params, # FYI: this is a dict
                        # metrics:
                        "accuracy": results.accy,
                        "f1_macro": results.f1_macro,
                        "f1_weighted": results.f1_weighted,
                        "roc_auc_score": results.roc_auc_score,
                    }
                    records.append(record)
                except FileNotFoundError as err:
                    print("MISSING:", results_filepath.replace(RESULTS_DIRPATH, ""))



    df = DataFrame(records)
    print(df.shape)
    df.sort_values(by="roc_auc_score", ascending=False, inplace=True)
    print(df.head(15))

    csv_filepath = os.path.join(RESULTS_DIRPATH, "reduced_classification", "all_results.csv")
    df.to_csv(csv_filepath, index=False)
