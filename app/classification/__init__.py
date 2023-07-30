
import os
import json

from app import RESULTS_DIRPATH

CLASSIFICATION_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "classification")


BOT_CLASSES_MAP = {True:"Bot", False:"Human"}
OPINION_CLASSES_MAP = {0:"Anti-Trump", 1:"Pro-Trump"}
CLASSES_MAP = {
    "is_bot": BOT_CLASSES_MAP,
    "is_bom_overall": BOT_CLASSES_MAP,
    "is_bom_astroturf": BOT_CLASSES_MAP,
    "opinion_community":OPINION_CLASSES_MAP
}


def save_results_json(results, json_filepath):
    with open(json_filepath, "w") as json_file:
        json.dump(results, json_file, indent=4)
