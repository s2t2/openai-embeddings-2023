import os
from functools import cached_property
from pandas import read_csv

from app.dataset import Dataset
from app.reduction.pipeline import REDUCTION_RESULTS_DIRPATH

FORCE_RECOMPILE = bool(os.getenv("FORCE_RECOMPILE", default="false") == "true")


REDUCED_DATASET_PATH = os.path.join(REDUCTION_RESULTS_DIRPATH, "botometer_sample_openai_tweet_embeddings_reduced_20230813.csv.gz")
REDUCTIONS = [
    ("pca", 2), ("pca", 3), ("pca", 7),
    ("tsne", 2), ("tsne", 3), ("tsne", 4),
    ("umap", 2), ("umap", 3),
]

def feature_colnames(reducer_name="pca", n_components=2):
    return [f"{reducer_name}_{n_components}_component_{i}" for i in range(1, n_components+1)]


class ReducedDataset(Dataset):
    """An enhanced version of the original dataset, plus embeddings from dimensionality reduction."""

    def __init__(self, reducer_name="pca", n_components=2):
        super().__init__(csv_filepath=REDUCED_DATASET_PATH)

        # for recompiling dataset (all methods):
        self.force_recompile = FORCE_RECOMPILE
        self.results_csv_filepaths = [os.path.join(REDUCTION_RESULTS_DIRPATH, f"{m}_{n}_embeddings.csv") for m, n in REDUCTIONS]

        # for analysis (single method):
        self.reducer_name = reducer_name
        self.reducer_type = {"pca": "PCA", "tsne": "T-SNE", "umap": "UMAP"}[self.reducer_name]
        self.n_components = n_components
        #self.title = f"OpenAI Embeddings * {self.reducer_type}-{self.n_components}"
        #self.chart_title = f"OpenAI Embeddings + {self.reducer_title}-{self.n_components}"

        self.feature_cols = feature_colnames(self.reducer_name, self.n_components)


    @cached_property
    def df(self):
        """Override parent method, compile dataset from reduction results."""
        if os.path.isfile(self.csv_filepath) and not self.force_recompile:
            print("LOADING EXISTING DATASET FROM FILE...")
            return read_csv(self.csv_filepath)
        else:
            print("COMPILING DATASET FROM RESULTS FILES...")
            ds = Dataset()
            df = ds.df
            # merge original dataset with reduction results files:
            for results_csv_filepath in self.reduction_results_csv_filepaths:
                embeddings_df = read_csv(results_csv_filepath)
                df = df.merge(embeddings_df, left_on="user_id", right_on="user_id")
            # write dataset (for faster loading later):
            df.to_csv(self.csv_filepath, index=False)
            return df


    @cached_property
    def x(self):
        """Override parent method, use feature cols for the given reduction method and n components."""
        return self.df[self.feature_cols].copy()


if __name__ == "__main__":

    ds = ReducedDataset()

    df = ds.df
    print(df.shape)
    print(df.columns)
