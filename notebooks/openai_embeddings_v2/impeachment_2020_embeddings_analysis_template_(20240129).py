# -*- coding: utf-8 -*-
"""Impeachment 2020 Embeddings Analysis Template (20240129)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dAlLxG-SbQNzBVLyD84a9x_6xlBUPQjQ

We fetched user-level and tweet-level OpenAI embeddings and stored on BQ, and copied the data to CSV files on Drive.

This notebook provides an example of how to load those CSV files. Feel free to make a copy of this notebook and perform your own analyses.

## Setup

### Google Drive
"""

import os
from google.colab import drive

drive.mount('/content/drive')
print(os.getcwd(), os.listdir(os.getcwd()))

# you might need to create a google drive SHORTCUT that has this same path
# ... or update the path to use your own google drive organization
#DIRPATH = '/content/drive/MyDrive/Research/Disinfo Research Shared 2022'
#DIRPATH = '/content/drive/MyDrive/Research/DS Research Shared 2023'
DIRPATH = '/content/drive/MyDrive/Research/DS Research Shared 2024'

print(DIRPATH)
os.path.isdir(DIRPATH)

"""New project-based directory structure for 2024:

https://drive.google.com/drive/folders/1SuXkqVT400uZ2OYFGGV8SYBf7NhtBo5k?usp=drive_link
"""

DATA_DIRPATH = os.path.join(DIRPATH, "projects", "Impeachment 2020 Embeddings", "data")
os.path.isdir(DATA_DIRPATH)

os.listdir(DATA_DIRPATH)

"""The "unpacked" versions have a column per embedding, and are generally easier to work with.

The files we will be working with are:
  +  "botometer_sample_max_50_openai_user_embeddings_unpacked.csv.gz" and
  + "botometer_sample_max_50_openai_status_embeddings_v3_unpacked.parquet.gzip".

## User Embeddings

7566 users

Loading CSV from drive:
"""

from pandas import read_csv

csv_filepath = os.path.join(DATA_DIRPATH, "botometer_sample_max_50_openai_user_embeddings_unpacked.csv.gz")
users_df = read_csv(csv_filepath, compression="gzip")
print(users_df.shape)
print(users_df.columns)
users_df.head()

users_df["user_id"].nunique()

users_df["is_bot"].value_counts()

users_df["opinion_community"].value_counts()

users_df["avg_fact_score"].info()

from pandas import isnull

def add_labels(users_df):
    # APPLY SAME LABELS AS THE ORIGINAL SOURCE CODE
    # https://github.com/s2t2/openai-embeddings-2023/blob/1b8372dd36982009df5d4a80871f4c182ada743d/notebooks/2_embeddings_data_export.py#L51
    # https://github.com/s2t2/openai-embeddings-2023/blob/main/app/dataset.py#L37-L64

    # labels:
    users_df["opinion_label"] = users_df["opinion_community"].map({0:"Anti-Trump", 1:"Pro-Trump"})
    users_df["bot_label"] = users_df["is_bot"].map({True:"Bot", False:"Human"})
    users_df["fourway_label"] = users_df["opinion_label"] + " " + users_df["bot_label"]

    # language toxicity scores (0 low - 1 high)
    toxic_threshold = 0.1
    users_df["is_toxic"] = users_df["avg_toxicity"] >= toxic_threshold
    users_df["is_toxic"] = users_df["is_toxic"].map({True: 1, False :0 })
    users_df["toxic_label"] = users_df["is_toxic"].map({1: "Toxic", 0 :"Normal" })

    # fact check / media quality scores (1 low - 5 high)
    fact_threshold = 3.0
    users_df["is_factual"] = users_df["avg_fact_score"].apply(lambda score: score if isnull(score) else score >= fact_threshold)

    # botometer binary and labels:
    users_df["is_bom_overall"] = users_df["bom_overall"].round()
    users_df["is_bom_astroturf"] = users_df["bom_astroturf"].round()
    users_df["bom_overall_label"] = users_df["is_bom_overall"].map({1:"Bot", 0:"Human"})
    users_df["bom_astroturf_label"] = users_df["is_bom_astroturf"].map({1:"Bot", 0:"Human"})
    users_df["bom_overall_fourway_label"] = users_df["opinion_label"] + " " + users_df["bom_overall_label"]
    users_df["bom_astroturf_fourway_label"] = users_df["opinion_label"] + " " + users_df["bom_astroturf_label"]

    return users_df


users_df = add_labels(users_df)
print(users_df.shape)
print(users_df.columns.tolist())
users_df.head()

users_df["is_factual"].value_counts()

users_df["is_toxic"].value_counts()

users_df["bot_label"].value_counts()

users_df["opinion_label"].value_counts()

users_df["fourway_label"].value_counts()

"""## Tweet Embeddings

183K statuses:
"""

from pandas import read_parquet

pq_filepath = os.path.join(DATA_DIRPATH, "botometer_sample_max_50_openai_status_embeddings_v3_unpacked.parquet.gzip")
statuses_df = read_parquet(pq_filepath)
print(statuses_df.shape)
print(statuses_df.columns)
statuses_df.head()

statuses_df["user_id"].nunique()

