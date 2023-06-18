# -*- coding: utf-8 -*-
"""OpenAI User Embeddings - Dimensionality Reduction - 20230617

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16MECGJrGdgk8kjRD4mzW1TQguCuS0225

## Setup

### Google Drive
"""

import os
from google.colab import drive

drive.mount('/content/drive')
print(os.getcwd(), os.listdir(os.getcwd())) #> 'content', ['.config', 'drive', 'sample_data']

# you might need to create a google drive SHORTCUT that has this same path
# ... or update the path to use your own google drive organization
DATA_DIR = '/content/drive/MyDrive/Research/DS Research Shared 2023/data/impeachment_2020'
print(DATA_DIR)
assert os.path.isdir(DATA_DIR)

#users_sample_csv_filepath = os.path.join(DATA_DIR, "users_sample_by_account_type_v2_and_their_tweets.csv")
#assert os.path.isfile(users_sample_csv_filepath)

MODEL_ID = "text-embedding-ada-002"

embeddings_csv_filepath = os.path.join(DATA_DIR, MODEL_ID, "users_sample_openai_embeddings.csv")
assert os.path.isfile(embeddings_csv_filepath)

"""### Packages"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install umap-learn[plot]

# Commented out IPython magic to ensure Python compatibility.
# # https://www.pauldesalvo.com/how-to-download-plotly-express-charts-as-images-in-google-colab/
# %%capture
# !pip install kaleido
# !pip install plotly>=4.0.0
# !wget https://github.com/plotly/orca/releases/download/v1.2.1/orca-1.2.1-x86_64.AppImage -O /usr/local/bin/orca
# !chmod +x /usr/local/bin/orca
# !apt-get install xvfb libgtk2.0-0 libgconf-2-4

"""## Load Embeddings"""

from pandas import read_csv

df = read_csv(embeddings_csv_filepath)
df.drop(columns=["user_id.1"], inplace=True)
#df.index = df["user_id"]
df.head()

"""### User Labels"""

df["opinion_label"] = df["opinion_community"].map({0:"Anti-Trump", 1:"Pro-Trump"})
df["bot_label"] = df["is_bot"].map({True:"Bot", False:"Human"})
df["q_label"] = df["is_q"].map({True:"Q-anon", False:"Normal"})

df["group_label"] = df["opinion_label"] + " " + df["q_label"] + " " + df["bot_label"]
df["group_label"].value_counts()



#GREY = "#ccc"
#PURPLE = "#7E57C2"

# light --> dark
#BLUES = ["#3498DB", "#2E86C1", "#2874A6"]
#REDS = ["#D98880", "#E6B0AA", "#C0392B", "#B03A2E", "#922B21"]

# colorbrewer scales
BLUES = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
REDS = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
PURPLES = ['#fcfbfd', '#efedf5', '#dadaeb', '#bcbddc', '#9e9ac8', '#807dba', '#6a51a3', '#54278f', '#3f007d']
GREYS = ['#ffffff', '#f0f0f0', '#d9d9d9', '#bdbdbd', '#969696', '#737373', '#525252', '#252525', '#000000']
#GREENS = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
#ORANGES = ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']


OPINION_COLORS_MAP = {"Anti-Trump": BLUES[5], "Pro-Trump": REDS[5]}
BOT_COLORS_MAP = {"Human": GREYS[3], "Bot": PURPLES[6]}
Q_COLORS_MAP = {"Normal":GREYS[3], "Q-anon": REDS[6]}

GROUP_COLORS_MAP = {
    "Anti-Trump Normal Human": BLUES[3],
    "Anti-Trump Normal Bot": BLUES[6],

    "Pro-Trump Normal Human": REDS[2],
    "Pro-Trump Normal Bot": REDS[3],

    "Pro-Trump Q-anon Human": REDS[6],
    "Pro-Trump Q-anon Bot": REDS[7],
}
#df["group_color"] = df["group_label"].map(GROUP_COLORS_MAP)



len(df)

"""### Unpack Embeddings

The embeddings happen to be stored as a JSON string, so we'll need to convert that single column into a column per value in the embeddings array. We'll get 1536 columns back.
"""

import json

def unpack(embeddings_str):
    # idempotence check
    if isinstance(embeddings_str, str):
        return json.loads(embeddings_str)
    else:
        return embeddings_str


df["tweet_embeddings"] = df["tweet_embeddings"].apply(unpack)
df["profile_embeddings"] = df["profile_embeddings"].apply(unpack)

type(df["tweet_embeddings"][0])
len(df["tweet_embeddings"][0]) #> 1536

"""These datasets have a column per embedding, and some user label columns."""

from pandas import DataFrame

tweet_embeddings = DataFrame(df["tweet_embeddings"].values.tolist())
profile_embeddings = DataFrame(df["profile_embeddings"].values.tolist())

LABEL_COLS = ["user_id", #"created_on", "screen_name_count", "status_count", "rt_count", "rt_pct", 
    "opinion_community", "is_bot", "is_q",
    # engineered labels:
    "opinion_label", "bot_label", "q_label", 
    "group_label" #, "group_color"
]
tweets_df = df[LABEL_COLS].merge(tweet_embeddings, left_index=True, right_index=True)
profiles_df = df[LABEL_COLS].merge(profile_embeddings, left_index=True, right_index=True)
profiles_df

"""## Dimensionality Reduction"""

import os
import numpy as np
from functools import cached_property

from pandas import DataFrame
from sklearn.preprocessing import scale #, StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import plotly.express as px

FIG_SHOW = False
FIG_SAVE = False

class ReductionPipeline:
    def __init__(self, df, label_cols, x_scale=True, reducer_type="PCA", n_components=2, labels_df=None, results_dirname="reduction_results"):
        """Params 
            df: a DataFrame with all the feature columns, plus some label columns 
        
            reducer_type: one of "PCA", "T-SNE", "UMAP"

            label_cols: list of strings: the columns you want to use for labeling / segmenting. 
                choose all column names except for the features.
        """
        self.df = df

        self.label_cols = label_cols
        self.labels_df = self.df[self.label_cols]
        self.x = self.df.drop(columns=self.label_cols)
        #print("X:", self.x.shape)
        
        #self.y = self.df[y_col]
        #print("Y:", len(self.y))
        
        self.x_scale = x_scale
        self.reducer_type = reducer_type
        self.n_components = n_components

        self.reducer_name = {"PCA": "pca", "T-SNE": "tsne", "UMAP": "umap"}[self.reducer_type]

        self.results_dirname = results_dirname

        self.reducer = None
        self.embeddings = None
        self.embeddings_df = None
        self.loadings = None
        self.loadings_df = None


    @cached_property
    def feature_names(self):
        # returns strings because PCA get_feature_names_out only works with string feature names
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        return [str(colname) for colname in self.x.columns.tolist()]

    @cached_property
    def component_names(self):
        return [f"component_{i}" for i in range(1, self.n_components+1)]

    @cached_property
    def x_scaled(self):
        x = scale(self.x)
        df = DataFrame(x, columns=self.feature_names)
        df.index = self.x.index
        return df

    def perform(self):
        if self.x_scale:
            x = self.x_scaled
        else:
            x = self.x

        if self.reducer_type == "PCA":
            self.reducer = PCA(n_components=self.n_components, random_state=99)
        elif self.reducer_type == "T-SNE":
            self.reducer = TSNE(n_components=self.n_components, random_state=99)
        elif self.reducer_type == "UMAP":
            self.reducer = UMAP(n_components=self.n_components, random_state=99)

        self.embeddings = self.reducer.fit_transform(x)
        #print("EMBEDDINGS:", type(self.embeddings), self.embeddings.shape)
        self.embeddings_df = DataFrame(self.embeddings, columns=self.component_names)
        self.embeddings_df = self.embeddings_df.merge(self.labels_df, left_index=True, right_index=True)

        # EXPLAINABILITY:
        if self.reducer_type == "PCA":
            #print("EXPLAINED VARIANCE RATIO:", self.reducer.explained_variance_ratio_)
            #print("SINGULAR VALS:", self.reducer.singular_values_)

            self.loadings = self.reducer.components_.T * np.sqrt(self.reducer.explained_variance_)
            #print("LOADINGS...", type(self.loadings), self.loadings.shape)
            self.loadings_df = DataFrame(self.loadings, columns=self.component_names)
            self.loadings_df.index = self.reducer.feature_names_in_

            # these represent the absolute magnitude of importances, not direction up or down
            self.feature_importances = {}
            for component_name in self.component_names:
                top_feature_names = self.loadings_df.abs().sort_values(by=[component_name], ascending=False).head(10)[component_name]
                self.feature_importances[component_name] = top_feature_names.to_dict()

        elif self.reducer_type == "T-SNE":
            print("K-L DIVERGENCE:", self.reducer.kl_divergence_)



    def results_dirpath(self, groupby):
        dirpath = os.path.join(self.results_dirname, groupby)  # f"groupby_{groupby}"
        os.makedirs(dirpath, exist_ok=True)
        return dirpath


    def embeddings_png_filepath(self, groupby):
        return os.path.join(self.results_dirpath(groupby), f"{self.reducer_name}_{self.n_components}.png")


    def embeddings_html_filepath(self, groupby):
        return os.path.join(self.results_dirpath(groupby), f"{self.reducer_name}_{self.n_components}.html")


    def centroids_png_filepath(self, groupby):
        return os.path.join(self.results_dirpath(groupby), f"{self.reducer_name}_{self.n_components}_centroids.png")


    def centroids_html_filepath(self, groupby):
        return os.path.join(self.results_dirpath(groupby), f"{self.reducer_name}_{self.n_components}_centroids.html")


    def plot_embeddings(self, height=500, fig_show=FIG_SHOW, fig_save=FIG_SAVE, title=None, subtitle=None, color=None, color_map=None):
        title = title or f"Dimensionality Reduction ({self.reducer_type} n_components={self.n_components})"
        if subtitle:
            title = title + f"<br><sup>{subtitle}</sup>"

        chart_params = dict(x="component_1", y="component_2", height=height,
            title=title, hover_data=self.label_cols
        )
        if color:
            chart_params["color"] = color
        if color_map:
            chart_params["color_discrete_map"] = color_map

        fig = None
        if self.n_components == 2:
            fig = px.scatter(self.embeddings_df, **chart_params)
        elif self.n_components ==3:
            chart_params["z"] = "component_3"
            fig = px.scatter_3d(self.embeddings_df, **chart_params)

        if fig and fig_show:
            fig.show()

        if fig and fig_save:
            #fig.write_image(self.embeddings_png_filepath(color))
            fig.write_html(self.embeddings_html_filepath(color))

        return fig


    def plot_embedding_centroids(self, groupby_col, height=500, fig_show=FIG_SHOW, fig_save=FIG_SAVE, title=None, subtitle=None, color_map=None):
        title = title or f"Dimensionality Reduction ({self.reducer_type} n_components={self.n_components}) Centroids"
        if subtitle:
            title = title + f"<br><sup>{subtitle}</sup>"

        chart_params = dict(x="component_1", y="component_2", height=height,
            title=title, #hover_data=self.label_cols,
            color=groupby_col, text=groupby_col              
        )
        if color_map:
            chart_params["color_discrete_map"] = color_map

        agg_params = {"component_1": "mean", "component_2": "mean"}

        fig = None
        if self.n_components == 2:
            centroids = self.embeddings_df.groupby(groupby_col).agg(agg_params)
            centroids[groupby_col] = centroids.index
            fig = px.scatter(centroids, **chart_params)

        elif self.n_components == 3:
            chart_params["z"] = "component_3"
            agg_params["component_3"] = "mean"
            centroids = self.embeddings_df.groupby(groupby_col).agg(agg_params)
            centroids[groupby_col] = centroids.index
            fig = px.scatter_3d(centroids, **chart_params)

        if fig:
            fig.update_traces(textposition='top center')

        if fig and fig_show:
            fig.show()

        if fig and fig_save:
            fig.write_html(self.centroids_html_filepath(groupby_col))

        return fig

"""### Save Figures"""

# you might need to create a google drive SHORTCUT that has this same path
# ... or update the path to use your own google drive organization
RESULTS_DIR = '/content/drive/MyDrive/Research/DS Research Shared 2023/users/mjr300/Impeachment 2020/reduction_results'
print(RESULTS_DIR)
os.makedirs(RESULTS_DIR, exist_ok=True)
assert os.path.isdir(RESULTS_DIR)

for n_components in [2,3]:

    for reducer_type in ["PCA", "T-SNE", "UMAP"]:
        print("---------------")
        print(reducer_type, n_components)

        results_dirname = os.path.join(RESULTS_DIR, "profiles")
        profiles_pipeline = ReductionPipeline(df=profiles_df, label_cols=LABEL_COLS, reducer_type=reducer_type, results_dirname=results_dirname, n_components=n_components)
        profiles_pipeline.perform()
        subtitle = "User Profile Embeddings"
        profiles_pipeline.plot_embeddings(fig_show=False,           fig_save=True, color="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
        profiles_pipeline.plot_embeddings(fig_show=False,           fig_save=True, color="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
        profiles_pipeline.plot_embeddings(fig_show=False,           fig_save=True, color="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
        profiles_pipeline.plot_embeddings(fig_show=False,           fig_save=True, color="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)
        profiles_pipeline.plot_embedding_centroids(fig_show=False,  fig_save=True, groupby_col="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
        profiles_pipeline.plot_embedding_centroids(fig_show=False,  fig_save=True, groupby_col="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
        profiles_pipeline.plot_embedding_centroids(fig_show=False,  fig_save=True, groupby_col="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
        profiles_pipeline.plot_embedding_centroids(fig_show=False,  fig_save=True, groupby_col="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

        results_dirname = os.path.join(RESULTS_DIR, "tweets")
        tweets_pipeline = ReductionPipeline(df=tweets_df, label_cols=LABEL_COLS, reducer_type=reducer_type, results_dirname=results_dirname, n_components=n_components)
        tweets_pipeline.perform()
        subtitle = "User Tweet Embeddings"
        tweets_pipeline.plot_embeddings(fig_show=False,          fig_save=True, color="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
        tweets_pipeline.plot_embeddings(fig_show=False,          fig_save=True, color="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
        tweets_pipeline.plot_embeddings(fig_show=False,          fig_save=True, color="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
        tweets_pipeline.plot_embeddings(fig_show=False,          fig_save=True, color="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)
        tweets_pipeline.plot_embedding_centroids(fig_show=False, fig_save=True, groupby_col="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
        tweets_pipeline.plot_embedding_centroids(fig_show=False, fig_save=True, groupby_col="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
        tweets_pipeline.plot_embedding_centroids(fig_show=False, fig_save=True, groupby_col="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
        tweets_pipeline.plot_embedding_centroids(fig_show=False, fig_save=True, groupby_col="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

"""### PCA Tuner"""

class PCATuner:

    def __init__(self, df, label_cols=LABEL_COLS, results_dirname="results"):
        self.df = df
        self.label_cols = label_cols
        self.feature_names = self.df.drop(columns=self.label_cols).columns.tolist()

        self.results_dirname = results_dirname
        self.results = None
        self.results_df = None


    def perform(self, components_limit=50):
        self.results = []

        components_range = range(1, len(self.feature_names)+1)
        if components_limit:
            components_range = components_range[0:components_limit]

        for n_components in components_range:
            pipeline = ReductionPipeline(self.df, label_cols=self.label_cols, 
                                         reducer_type="PCA", n_components=n_components)
            pipeline.perform()

            pca = pipeline.reducer
            self.results.append({
                "n_components": n_components,
                "explained_variance": pca.explained_variance_ratio_.sum(),
                "eigenvals": pca.explained_variance_, # number of vals depend on n components
                #"loadings": loadings,
                #"embeddings": embeddings
            })
        self.results_df = DataFrame(self.results)
        #print(self.results_df[["n_components", "explained_variance"]].head())





    @property
    def results_dirpath(self):
        #dirpath = os.path.join(RESULTS_DIRPATH, "youtube", f"length_{self.track_length}_mfcc_{self.n_mfcc}")
        dirpath = self.results_dirname # "results" # colab
        os.makedirs(dirpath, exist_ok=True)
        return dirpath


    def plot_explained_variance(self, height=500, fig_show=FIG_SHOW, fig_save=FIG_SAVE, subtitle=None, log_y=False):
        title = f"Total Explained Variance by Number of Components (PCA)"
        if subtitle:
            title = title + f"<br><sup>{subtitle}</sup>"
        
        chart_opts = dict(x="n_components", y="explained_variance",
                title=title, height=height,
                markers="line+point", color_discrete_sequence=["steelblue"],
        )
        if log_y:
            chart_opts["log_y"] = True # range_y=[0,1] # range_x=[1,100000], "]

        fig = px.line(self.results_df, **chart_opts)
        if fig_show:
            fig.show()

        if fig_save:
            image_filepath = os.path.join(self.results_dirpath, "pca-explained-variance.png")
            fig.write_image(image_filepath)
        #return fig


    def plot_scree(self, height=500, fig_show=FIG_SHOW, fig_save=FIG_SAVE, subtitle=None, log_y=False):
        eigenvals = self.results_df.sort_values(by=["n_components"], ascending=False).iloc[0]["eigenvals"]
        print("EIGENVALS:", eigenvals)

        component_numbers = list(range(1, len(self.results_df)+1))
        print("COMPONENT NUMBERS:", component_numbers)

        title=f"Scree Plot of Eigenvalues by Component (PCA)"
        if subtitle:
            title = title + f"<br><sup>{subtitle}</sup>"

        fig = px.line(x=component_numbers, y=eigenvals,
                title=title, height=height,
                labels={"x": "Component Number", "y": "Eigenvalue"},
                markers="line+point", color_discrete_sequence=["steelblue"],
                log_y=log_y
        )
        if fig_show:
            fig.show()

        if fig_save:
            image_filepath = os.path.join(self.results_dirpath, "pca-scree.png")
            fig.write_image(image_filepath)
        #return fig

"""#### Profiles"""

results_dirname = os.path.join(RESULTS_DIR, "profiles")
profile_tuner = PCATuner(df=profiles_df, label_cols=LABEL_COLS, results_dirname=results_dirname)
profile_tuner.perform(components_limit=100)
subtitle = "User Profile Embeddings"
profile_tuner.plot_explained_variance(fig_show=True, fig_save=True, subtitle=subtitle)
profile_tuner.plot_scree(fig_show=True, fig_save=True, subtitle=subtitle)

"""#### Tweets"""

results_dirname = os.path.join(RESULTS_DIR, "tweets")
tweet_tuner = PCATuner(df=tweets_df, label_cols=LABEL_COLS, results_dirname=results_dirname)
tweet_tuner.perform(components_limit=100)
subtitle = "User Tweet Embeddings"
tweet_tuner.plot_explained_variance(fig_show=True, fig_save=True, subtitle=subtitle)
tweet_tuner.plot_scree(fig_show=True, fig_save=True, subtitle=subtitle)



"""### PCA (n=2)

#### Profiles
"""

profile_pca = ReductionPipeline(df=profiles_df, label_cols=LABEL_COLS)
profile_pca.perform()

subtitle = "User Profile Embeddings"
profile_pca.plot_embeddings(fig_show=True, color="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
profile_pca.plot_embeddings(fig_show=True, color="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
profile_pca.plot_embeddings(fig_show=True, color="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
profile_pca.plot_embeddings(fig_show=False, color="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

subtitle = "User Profile Embeddings"
#profile_pca.plot_embedding_centroids(fig_show=True, groupby_col="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
#profile_pca.plot_embedding_centroids(fig_show=True, groupby_col="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
#profile_pca.plot_embedding_centroids(fig_show=True, groupby_col="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
profile_pca.plot_embedding_centroids(fig_show=False, groupby_col="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

"""#### Tweets"""

tweets_pca = ReductionPipeline(df=tweets_df, label_cols=LABEL_COLS)
tweets_pca.perform()

subtitle = "User Tweet Embeddings"
tweets_pca.plot_embeddings(fig_show=True, color="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
tweets_pca.plot_embeddings(fig_show=True, color="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
tweets_pca.plot_embeddings(fig_show=True, color="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
tweets_pca.plot_embeddings(fig_show=False, color="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

subtitle = "User Tweet Embeddings"
#tweets_pca.plot_embedding_centroids(fig_show=True, groupby_col="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
#tweets_pca.plot_embedding_centroids(fig_show=True, groupby_col="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
#tweets_pca.plot_embedding_centroids(fig_show=True, groupby_col="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
tweets_pca.plot_embedding_centroids(fig_show=False, groupby_col="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

"""### T-SNE (n=2)

#### Profiles
"""

#profile_tsne = ReductionPipeline(df=profiles_df, label_cols=LABEL_COLS, reducer_type="T-SNE")
#profile_tsne.perform()

#subtitle = "User Profile Embeddings"
#profile_tsne.plot_embeddings(fig_show=True, color="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
#profile_tsne.plot_embeddings(fig_show=True, color="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
#profile_tsne.plot_embeddings(fig_show=True, color="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
#profile_tsne.plot_embeddings(fig_show=False, color="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

#subtitle = "User Profile Embeddings"
##profile_tsne.plot_embedding_centroids(fig_show=True, groupby_col="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
##profile_tsne.plot_embedding_centroids(fig_show=True, groupby_col="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
##profile_tsne.plot_embedding_centroids(fig_show=True, groupby_col="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
#profile_tsne.plot_embedding_centroids(fig_show=False, groupby_col="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

"""#### Tweets"""

tweets_tsne = ReductionPipeline(df=tweets_df, label_cols=LABEL_COLS, reducer_type="T-SNE")
tweets_tsne.perform()

subtitle = "User Tweet Embeddings"
tweets_tsne.plot_embeddings(fig_show=True, color="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
tweets_tsne.plot_embeddings(fig_show=True, color="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
tweets_tsne.plot_embeddings(fig_show=True, color="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
tweets_tsne.plot_embeddings(fig_show=False, color="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

subtitle = "User Tweet Embeddings"
#tweets_tsne.plot_embedding_centroids(fig_show=True, groupby_col="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
#tweets_tsne.plot_embedding_centroids(fig_show=True, groupby_col="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
#tweets_tsne.plot_embedding_centroids(fig_show=True, groupby_col="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
tweets_tsne.plot_embedding_centroids(fig_show=False, groupby_col="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

"""### UMAP (n=2)

#### Profiles
"""

#profile_umap = ReductionPipeline(df=profiles_df, label_cols=LABEL_COLS, reducer_type="UMAP")
#profile_umap.perform()

#subtitle = "User Profile Embeddings"
#profile_umap.plot_embeddings(fig_show=True, color="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
#profile_umap.plot_embeddings(fig_show=True, color="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
#profile_umap.plot_embeddings(fig_show=True, color="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
#profile_umap.plot_embeddings(fig_show=False, color="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

#subtitle = "User Profile Embeddings"
##profile_umap.plot_embedding_centroids(fig_show=True, groupby_col="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
##profile_umap.plot_embedding_centroids(fig_show=True, groupby_col="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
##profile_umap.plot_embedding_centroids(fig_show=True, groupby_col="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
#profile_umap.plot_embedding_centroids(fig_show=False, groupby_col="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

"""#### Tweets"""

tweets_umap = ReductionPipeline(df=tweets_df, label_cols=LABEL_COLS, reducer_type="UMAP")
tweets_umap.perform()

#subtitle = "User Tweet Embeddings"
#tweets_umap.plot_embeddings(fig_show=True, color="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
#tweets_umap.plot_embeddings(fig_show=True, color="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
#tweets_umap.plot_embeddings(fig_show=True, color="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
#tweets_umap.plot_embeddings(fig_show=False, color="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)

subtitle = "User Tweet Embeddings"
#tweets_umap.plot_embedding_centroids(fig_show=True, groupby_col="opinion_label", subtitle=subtitle, color_map=OPINION_COLORS_MAP)
#tweets_umap.plot_embedding_centroids(fig_show=True, groupby_col="bot_label", subtitle=subtitle, color_map=BOT_COLORS_MAP)
#tweets_umap.plot_embedding_centroids(fig_show=True, groupby_col="q_label", subtitle=subtitle, color_map=Q_COLORS_MAP)
tweets_umap.plot_embedding_centroids(fig_show=False, groupby_col="group_label", subtitle=subtitle, color_map=GROUP_COLORS_MAP)