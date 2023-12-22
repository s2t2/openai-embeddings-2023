
import os

import plotly.express as px

from app import RESULTS_DIRPATH
from app.reduction.pipeline import ReductionPipeline, FIG_SAVE, FIG_SHOW
from app.reduction.tuner import ReductionTuner, MAX_COMPONENTS


class PCATuner(ReductionTuner):

    def __init__(self, ds=None, results_dirpath=RESULTS_DIRPATH, max_components=MAX_COMPONENTS):
        super().__init__(ds=ds, results_dirpath=results_dirpath, max_components=max_components,
                         reducer_type="PCA")

    #def perform(self):
    #    self.results = []
    #
    #    # if we have lots of columns / features, we might want to abbreviate the search space and override with a max value, otherwise search over all available features
    #    max_components = self.max_components or len(self.feature_names)
    #    # get the explained variance for each n up to the max number of components to search over
    #    for n_components in range(1, max_components+1):
    #
    #        pipeline = ReductionPipeline(df=self.df, label_cols=self.label_cols,
    #                                     reducer_type="PCA", n_components=n_components)
    #        pipeline.perform()
    #
    #        pca = pipeline.reducer
    #        self.results.append({
    #            "n_components": n_components,
    #            "explained_variance": pca.explained_variance_ratio_.sum(),
    #            "eigenvals": pca.explained_variance_, # number of vals depend on n components
    #            #"loadings": loadings,
    #            #"embeddings": embeddings
    #        })
    #    self.results_df = DataFrame(self.results)
    #    print(self.results_df[["n_components", "explained_variance"]].head())

    def collect_result(self, pipeline:ReductionPipeline, n_components:int):
        pca = pipeline.reducer
        self.results.append({
                "n_components": n_components,
                "explained_variance": pca.explained_variance_ratio_.sum(),
                "eigenvals": pca.explained_variance_, # number of vals depend on n components
                #"loadings": loadings,
                #"embeddings": embeddings
            })

    def print_results(self):
        print(self.results_df[["n_components", "explained_variance"]].head())

    def plot_explained_variance(self, height=500, fig_show=FIG_SHOW, fig_save=FIG_SAVE, subtitle=None, results_dirpath=None):
        title = f"Total Explained Variance by Number of Components (PCA)"
        subtitle = subtitle or f"Max Components: {self.max_components}"
        title += f"<br><sup>{subtitle}</sup>"

        fig = px.line(self.results_df, x="n_components", y="explained_variance",
                title=title, height=height,
                markers="line+point", color_discrete_sequence=["steelblue"]
        )
        if fig_show:
            fig.show()

        if fig_save:
            results_dirpath = results_dirpath or self.results_dirpath
            filestem = "pca-explained-variance" if not self.max_components else f"pca-explained-variance-{self.max_components}"
            image_filepath = os.path.join(results_dirpath, f"{filestem}.png")
            html_filepath = os.path.join(results_dirpath, f"{filestem}.html")
            fig.write_image(image_filepath)
            fig.write_html(html_filepath)
        #return fig


    def plot_scree(self, height=500, fig_show=FIG_SHOW, fig_save=FIG_SAVE, subtitle=None, results_dirpath=None):
        eigenvals = self.results_df.sort_values(by=["n_components"], ascending=False).iloc[0]["eigenvals"]
        print("EIGENVALS:", eigenvals)

        component_numbers = list(range(1, len(self.results_df)+1))
        print("COMPONENT NUMBERS:", component_numbers)

        title=f"Scree Plot of Eigenvalues by Component (PCA)"
        subtitle = subtitle or f"Max Components: {self.max_components}"
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
            filestem = "pca-scree" if not self.max_components else f"pca-scree-{self.max_components}"
            image_filepath = os.path.join(results_dirpath, f"{filestem}.png")
            html_filepath = os.path.join(results_dirpath, f"{filestem}.html")
            fig.write_image(image_filepath)
            fig.write_html(html_filepath)
        #return fig


if __name__ == "__main__":

    from app.dataset import Dataset

    ds = Dataset()

    tuner = PCATuner(df=ds.df, label_cols=ds.label_cols)
    tuner.perform()
    tuner.plot_explained_variance()
    tuner.plot_scree()
