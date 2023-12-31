
import os

from app import RESULTS_DIRPATH

CLASSIFICATION_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "classification")

Y_COLS_BINARY = [
    # binary classification with boolean classes:
    "is_bot",
    "opinion_community",
    #"is_q", # sample size is too small for meaningful analysis :-/
    "is_bom_overall", "is_bom_astroturf",
    "is_toxic",
    "is_factual",
]
Y_COLS_MULTICLASS = [
    # multiclass classification with categorical classes:
    "fourway_label", #"bom_overall_fourway_label", "bom_astroturf_fourway_label"
]
Y_COLS = Y_COLS_BINARY + Y_COLS_MULTICLASS

BOT_CLASSES_MAP = {True:"Bot", False:"Human"}
CLASSES_MAP = {
    "is_bot": BOT_CLASSES_MAP,
    "is_bom_overall": BOT_CLASSES_MAP,
    "is_bom_astroturf": BOT_CLASSES_MAP,
    "opinion_community": {0:"Anti-Trump", 1:"Pro-Trump"},
    "is_toxic": {0: "Normal", 1: "Toxic"},
    "is_factual": {0: "Low Quality", 1: "High Quality"},
}



def class_labels(y_col, class_names):
    # apply custom labels for binary values, to make chart easier to read
    # keep order the same as they were passed in
    # basically convert from classifier.classes_
    if y_col in CLASSES_MAP.keys():
        classes_map = CLASSES_MAP[y_col]
        class_names = [classes_map[val] for val in class_names]
    return class_names
