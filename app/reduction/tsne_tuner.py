
import os

import plotly.express as px

from app import RESULTS_DIRPATH
from app.reduction.pipeline import ReductionPipeline, FIG_SAVE, FIG_SHOW
from app.reduction.tuner import ReductionTuner, MAX_COMPONENTS



class TSNETuner(ReductionTuner):
    """FYI: T-SNE gets slow when n_components >= 4

    K-L divergence measures difference between two distributions, where the
        value is 0 when the two distributions are equal.

    """

    def __init__(self, df, label_cols, results_dirpath=RESULTS_DIRPATH, max_components=MAX_COMPONENTS):
        super().__init__(df=df, label_cols=label_cols, results_dirpath=results_dirpath, max_components=max_components,
                         reducer_type="T-SNE")

    #def perform(self):
    #    self.results = []
    #
    #    # if we have lots of columns / features, we might want to abbreviate the search space and override with a max value, otherwise search over all available features
    #    max_components = self.max_components or len(self.feature_names)
    #    # get the explained variance for each n up to the max number of components to search over
    #    for n_components in range(1, max_components+1):
    #        # we need to use PCA specifically because unlike other methods it gives us the explainability metrics
    #        pipeline = ReductionPipeline(df=self.df, label_cols=self.label_cols,
    #                                     reducer_type="T-SNE", n_components=n_components)
    #        pipeline.perform()
    #
    #        self.results.append({
    #            "n_components": n_components,
    #            "kl_divergence": pipeline.reducer.kl_divergence_,
    #        })
    #    self.results_df = DataFrame(self.results)
    #    print(self.results_df[["n_components", "kl_divergence"]].head())

    def collect_result(self, pipeline:ReductionPipeline, n_components:int):
        self.results.append({
                "n_components": n_components,
                "kl_divergence": pipeline.reducer.kl_divergence_,
            })

    def print_results(self):
        print(self.results_df[["n_components", "kl_divergence"]].head())

    def plot_kl_divergence(self, height=500, fig_show=FIG_SHOW, fig_save=FIG_SAVE, subtitle=None, results_dirpath=None):
        title = f"K-L Divergence by Number of Components (T-SNE)"
        subtitle = subtitle or f"Max Components: {self.max_components}"
        title += f"<br><sup>{subtitle}</sup>"

        fig = px.line(self.results_df, x="n_components", y="kl_divergence",
                title=title, height=height,
                markers="line+point", color_discrete_sequence=["steelblue"]
        )
        if fig_show:
            fig.show()

        if fig_save:
            results_dirpath = results_dirpath or self.results_dirpath
            filestem = "tsne-kl-divergence" if not self.max_components else f"tsne-kl-divergence-{self.max_components}"
            image_filepath = os.path.join(results_dirpath, f"{filestem}.png")
            html_filepath = os.path.join(results_dirpath, f"{filestem}.html")
            fig.write_image(image_filepath)
            fig.write_html(html_filepath)
        #return fig




if __name__ == "__main__":

    from app.dataset import Dataset

    ds = Dataset()

    tuner = TSNETuner(df=ds.df, label_cols=ds.label_cols)
    tuner.perform()
    tuner.plot_kl_divergence()
