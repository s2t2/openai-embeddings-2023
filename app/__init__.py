

import os
import json

DATA_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIRPATH = os.path.join(os.path.dirname(__file__), "..", "results")



def save_results_json(results, json_filepath):
    with open(json_filepath, "w") as json_file:
        json.dump(results, json_file, indent=4)
