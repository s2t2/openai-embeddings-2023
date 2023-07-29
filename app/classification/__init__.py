
import os
import json

from app import RESULTS_DIRPATH

CLASSIFICATION_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "classification")



def save_results_json(results, json_filepath):
    with open(json_filepath, "w") as json_file:
        json.dump(results, json_file, indent=4)
