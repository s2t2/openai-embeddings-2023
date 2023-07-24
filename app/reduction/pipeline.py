
import warnings
# https://github.com/slundberg/shap/issues/2909
# suppress umap warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
# https://discuss.python.org/t/how-to-silence-pkg-resources-warnings/28629/7
# suppress warnings.warn("pkg_resources is deprecated as an API", DeprecationWarning)
warnings.simplefilter("ignore", DeprecationWarning)


import os
import numpy as np
from functools import cached_property

from pandas import DataFrame
from sklearn.preprocessing import scale #, StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import plotly.express as px

from app import RESULTS_DIRPATH
from app.dataset import Dataset

REDUCTION_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "reduction")

REDUCER_TYPE = os.getenv("REDUCER_TYPE", default="PCA") # "PCA", "T-SNE", "UMAP"
N_COMPONENTS = int(os.getenv("N_COMPONENTS", default="2"))
X_SCALE = bool(os.getenv("X_SCALE", default="true") == "true")
FIG_SHOW = bool(os.getenv("FIG_SHOW", default="false") == "true")
FIG_SAVE = bool(os.getenv("FIG_SAVE", default="false") == "true")

class ReductionPipeline:
    def __init__(self, ds=None, x_scale=X_SCALE,
                        reducer_type=REDUCER_TYPE, n_components=N_COMPONENTS,
                        results_dirpath=REDUCTION_RESULTS_DIRPATH):

        self.ds = ds or Dataset()
        self.labels_df = self.ds.labels
        #self.x = self.df.drop(columns=label_cols)

        self.x_scale = x_scale
        self.reducer_type = reducer_type
        self.n_components = n_components
        self.results_dirpath = results_dirpath

        if self.x_scale:
            self.x = self.ds.x_scaled
        else:
            self.x = self.ds.x
        #print("X:", self.x.shape)

        self.reducer_name = {"PCA": "pca", "T-SNE": "tsne", "UMAP": "umap"}[self.reducer_type]

        self.reducer = None
        self.embeddings = None
        self.embeddings_df = None
        self.loadings = None
        self.loadings_df = None


    @cached_property
    def feature_names(self):
        return self.x.columns.tolist()

    @cached_property
    def component_names(self):
        return [f"component_{i}" for i in range(1, self.n_components+1)]

    #@cached_property
    #def x_scaled(self):
    #    x = scale(self.x)
    #    df = DataFrame(x, columns=self.feature_names)
    #    df.index = self.x.index
    #    return df

    def perform(self):
        if self.reducer_type == "PCA":
            self.reducer = PCA(n_components=self.n_components, random_state=99)
        elif self.reducer_type == "T-SNE":
            # https://stackoverflow.com/questions/66592804/t-sne-can-not-convert-high-dimension-data-to-more-than-4-dimension
            # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
            # ValueError: 'n_components' should be inferior to 4 for the barnes_hut algorithm
            tsne_method = "exact" if self.n_components >= 4 else "barnes_hut"
            self.reducer = TSNE(n_components=self.n_components, random_state=99, method=tsne_method)
        elif self.reducer_type == "UMAP":
            self.reducer = UMAP(n_components=self.n_components, random_state=99)

        self.embeddings = self.reducer.fit_transform(self.x)
        print("EMBEDDINGS:", self.embeddings.shape)
        self.embeddings_df = DataFrame(self.embeddings, columns=self.component_names)
        self.embeddings_df = self.embeddings_df.merge(self.labels_df, left_index=True, right_index=True)

        # EXPLAINABILITY:
        if self.reducer_type == "PCA":
            print("EXPLAINED VARIANCE RATIO:", self.reducer.explained_variance_ratio_)
            print("SINGULAR VALS:", self.reducer.singular_values_)

            self.loadings = self.reducer.components_.T * np.sqrt(self.reducer.explained_variance_)
            print("LOADINGS...", self.loadings.shape)
            self.loadings_df = DataFrame(self.loadings, columns=self.component_names)
            self.loadings_df.index = self.reducer.feature_names_in_

            # these represent the absolute magnitude of importances, not direction up or down
            self.feature_importances = {}
            for component_name in self.component_names:
                top_feature_names = self.loadings_df.abs().sort_values(by=[component_name], ascending=False).head(10)[component_name]
                self.feature_importances[component_name] = top_feature_names.to_dict()

        elif self.reducer_type == "T-SNE":
            print("K-L DIVERGENCE:", self.reducer.kl_divergence_)


    def plot_embeddings(self, height=500, fig_show=FIG_SHOW, fig_save=FIG_SAVE, subtitle=None, color=None, color_map=None, category_orders=None, hover_data=None, results_dirpath=None):
        title = f"Dimensionality Reduction Results ({self.reducer_type} n_components={self.n_components})"
        if subtitle:
            title += f"<br><sup>{subtitle}</sup>"

        chart_params = dict(x="component_1", y="component_2",
            title=title, height=height,
            #color=color, #"artist_name",
            #hover_data=hover_data #["audio_filename", "track_number"]
        )
        if color:
            chart_params["color"] = color
        if color_map:
            chart_params["color_discrete_map"] = color_map
        if category_orders:
            chart_params["category_orders"] = category_orders
        if hover_data:
            chart_params["hover_data"] = hover_data

        if self.n_components == 2:
            fig = px.scatter(self.embeddings_df, **chart_params)
        elif self.n_components == 3:
            chart_params["z"] = "component_3"
            fig = px.scatter_3d(self.embeddings_df, **chart_params)
        else:
            return None

        if fig_show:
            fig.show()

        if fig_save:
            results_dirpath = results_dirpath or self.results_dirpath
            embeddings_html_filepath = os.path.join(results_dirpath, f"{self.reducer_name}_{self.n_components}.html")
            fig.write_html(embeddings_html_filepath)

        return fig


    def plot_centroids(self, groupby_col, height=500, fig_show=FIG_SHOW, fig_save=FIG_SAVE, title=None, subtitle=None, color_map=None, category_orders=None, results_dirpath=None):
        title = title or f"Dimensionality Reduction Centroids ({self.reducer_type} n_components={self.n_components})"
        if subtitle:
            title += f"<br><sup>{subtitle}</sup>"

        chart_params = dict(x="component_1", y="component_2", height=height,
            title=title, #hover_data=self.label_cols,
            color=groupby_col, text=groupby_col
        )
        if color_map:
            chart_params["color_discrete_map"] = color_map
        if category_orders:
            chart_params["category_orders"] = category_orders

        agg_params = {"component_1": "mean", "component_2": "mean"}

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
        else:
            return None

        fig.update_traces(textposition="top center")

        if fig_show:
            fig.show()

        if fig_save:
            results_dirpath = results_dirpath or self.results_dirpath
            #centroids_png_filepath = os.path.join(results_dirpath, f"{self.reducer_name}_{self.n_components}_centroids.png")
            centroids_html_filepath = os.path.join(results_dirpath, f"{self.reducer_name}_{self.n_components}_centroids.html")
            fig.write_html(centroids_html_filepath)

        return fig




if __name__ == "__main__":


    from app.colors import COLORS_MAP, CATEGORY_ORDERS

    pca_pipeline = ReductionPipeline()
    pca_pipeline.perform()

    for groupby_col in ["bot_label", "opinion_label", "fourway_label", "sixway_label", "bom_overall_label", "bom_astroturf_label"]:
        color_map = COLORS_MAP[groupby_col]
        category_orders = {groupby_col: CATEGORY_ORDERS[groupby_col]}

        results_dirpath = os.path.join(REDUCTION_RESULTS_DIRPATH, groupby_col)
        os.makedirs(results_dirpath, exist_ok=True)

        pca_pipeline.plot_embeddings(color=groupby_col, color_map=color_map,
                                     category_orders=category_orders,
                                #hover_data=["user_id", "bot_label"],
                                #fig_show=True, fig_save=True,
                                results_dirpath=results_dirpath
                                )

        pca_pipeline.plot_centroids(groupby_col=groupby_col, color_map=color_map,
                                    category_orders=category_orders,
                                #hover_data=["user_id", "bot_label"],
                                #fig_show=True, fig_save=True
                                results_dirpath=results_dirpath
                                )
