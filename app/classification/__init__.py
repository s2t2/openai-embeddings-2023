
import os
import json

from app import RESULTS_DIRPATH

CLASSIFICATION_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "classification")

Y_COLS = [
    # binary classification with boolean classes:
    "is_bot", "opinion_community", "is_bom_overall", "is_bom_astroturf",
    # multiclass classification with categorical classes:
    "fourway_label", "bom_overall_fourway_label", "bom_astroturf_fourway_label"
]

BOT_CLASSES_MAP = {True:"Bot", False:"Human"}
OPINION_CLASSES_MAP = {0:"Anti-Trump", 1:"Pro-Trump"}
CLASSES_MAP = {
    "is_bot": BOT_CLASSES_MAP,
    "is_bom_overall": BOT_CLASSES_MAP,
    "is_bom_astroturf": BOT_CLASSES_MAP,
    "opinion_community":OPINION_CLASSES_MAP
}


def class_labels(y_col, class_names):
    # apply custom labels for binary values, to make chart easier to read
    # keep order the same as they were passed in
    # basically convert from classifier.classes_
    if y_col in CLASSES_MAP.keys():
        classes_map = CLASSES_MAP[y_col]
        class_names = [classes_map[val] for val in class_names]
    return class_names


def save_results_json(results, json_filepath):
    with open(json_filepath, "w") as json_file:
        json.dump(results, json_file, indent=4)
