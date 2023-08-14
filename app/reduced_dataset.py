import os
from pandas import read_csv

from app.dataset import Dataset
from app.reduction.pipeline import REDUCTION_RESULTS_DIRPATH

FORCE_RECOMPILE = bool(os.getenv("FORCE_RECOMPILE", default="false") == "true")

#REDUCED_LABELS = [""]
#
#class ReducedDataset(Dataset):
#    """A super version of the original dataset, including reduction results."""
#
#    def __init__(self):
#        reduced_dataset_path = os.path.join(REDUCTION_RESULTS_DIRPATH, "reduced_embeddings_20230813.csv")
#        super().__init__(csv_filepath=reduced_dataset_path, label_cols=REDUCED_LABELS)
#
#        self.title = "Tweet Embeddings Dataset (Reduced)"
#
#
#
#    def df(self):
#

if __name__ == "__main__":

    reduced_dataset_path = os.path.join(REDUCTION_RESULTS_DIRPATH, "reduced_embeddings_20230813.csv")

    if os.path.isfile(reduced_dataset_path) and not FORCE_RECOMPILE:
        print("LOADING EXISTING DATASET FROM FILE...")
        df = read_csv(reduced_dataset_path)
    else:
        print("COMPILING DATASET FROM RESULTS FILES...")
        ds = Dataset()
        df = ds.df

        csv_filenames = [
            "pca_2_embeddings.csv",
            "pca_3_embeddings.csv",
            "pca_7_embeddings.csv",
            "tsne_2_embeddings.csv",
            "tsne_3_embeddings.csv",
            "tsne_4_embeddings.csv",
            "umap_2_embeddings.csv",
            "umap_3_embeddings.csv",
        ]
        for csv_filename in csv_filenames:
            csv_filepath = os.path.join(REDUCTION_RESULTS_DIRPATH, csv_filename)
            if os.path.isfile(csv_filepath):
                embeddings_df = read_csv(csv_filepath)
                df = df.merge(embeddings_df, left_on="user_id", right_on="user_id")
            else:
                print("--------------")
                print("OOPS, COULDN'T FIND FILE:", csv_filename)

        df.to_csv(reduced_dataset_path, index=False)

    print(df.shape)
    print(df.columns)
