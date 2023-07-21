
import os

from pandas import DataFrame
import plotly.express as px

from app import RESULTS_DIRPATH
from app.reduction.pipeline import ReductionPipeline, FIG_SAVE, FIG_SHOW


MAX_COMPONENTS = os.getenv("MAX_COMPONENTS")


class PCATuner:

    def __init__(self, df, label_cols=[], results_dirpath=RESULTS_DIRPATH, max_components=MAX_COMPONENTS):
        self.df = df

        self.label_cols = label_cols
        self.feature_names = self.df.drop(columns=self.label_cols).columns.tolist()

        if max_components:
            max_components = int(max_components)
        self.max_components = max_components

        self.results_dirpath = results_dirpath
        #os.makedirs(self.results_dirpath, exist_ok=True)

        self.results = None
        self.results_df = None


    def perform(self):
        self.results = []

        # if we have lots of columns / features, we might want to abbreviate the search space and override with a max value, otherwise search over all available features
        max_components = self.max_components or len(self.feature_names)
        # get the explained variance for each n up to the max number of components to search over
        for n_components in range(1, max_components+1):
            # we need to use PCA specifically because unlike other methods it gives us the explainability metrics
            pipeline = ReductionPipeline(df=self.df, label_cols=self.label_cols,
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
        print(self.results_df[["n_components", "explained_variance"]].head())







    def plot_explained_variance(self, height=500, fig_show=FIG_SHOW, fig_save=FIG_SAVE, subtitle=None, results_dirpath=None):
        title = f"Total Explained Variance by Number of Components (PCA)"
        if subtitle:
            title += f"<br><sup>{subtitle}</sup>"

        fig = px.line(self.results_df, x="n_components", y="explained_variance",
                title=title, height=height,
                markers="line+point", color_discrete_sequence=["steelblue"]
        )
        if fig_show:
            fig.show()

        if fig_save:
            results_dirpath = results_dirpath or self.results_dirpath
            image_filepath = os.path.join(results_dirpath, "pca-explained-variance.png")
            fig.write_image(image_filepath)
        #return fig


    def plot_scree(self, height=500, fig_show=FIG_SHOW, fig_save=FIG_SAVE, subtitle=None, results_dirpath=None):
        eigenvals = self.results_df.sort_values(by=["n_components"], ascending=False).iloc[0]["eigenvals"]
        print("EIGENVALS:", eigenvals)

        component_numbers = list(range(1, len(self.results_df)+1))
        print("COMPONENT NUMBERS:", component_numbers)

        title=f"Scree Plot of Eigenvalues by Component (PCA)"
        if subtitle:
            title += f"<br><sup>{subtitle}</sup>"

        fig = px.line(x=component_numbers, y=eigenvals,
                title=title, height=height,
                labels={"x": "Component Number", "y": "Eigenvalue"},
                markers="line+point", color_discrete_sequence=["steelblue"]
        )
        if fig_show:
            fig.show()

        if fig_save:
            results_dirpath = results_dirpath or self.results_dirpath
            image_filepath = os.path.join(results_dirpath, "pca-scree.png")
            fig.write_image(image_filepath)
        #return fig


if __name__ == "__main__":

    from app.dataset import Dataset

    ds = Dataset()

    tuner = PCATuner(df=ds.df, label_cols=ds.label_cols)
    tuner.perform()
    tuner.plot_explained_variance()
    tuner.plot_scree()
