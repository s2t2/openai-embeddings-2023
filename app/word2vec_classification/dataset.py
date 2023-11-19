
import os
from functools import cached_property
from pandas import read_csv

from app import DATA_DIRPATH
from app.dataset import Dataset
from app.word2vec_embeddings.pipeline import WORD2VEC_RESULTS_DIRPATH


WORD2VEC_EMBEDDINGS_CSV_FILEPATH = os.path.join(WORD2VEC_RESULTS_DIRPATH, "document_vectors.csv")

WORD2VEC_DATASET_PATH = os.path.join(DATA_DIRPATH, "word2vec", "botometer_sample_word2vec_embeddings_20230825.csv.gz")

class Word2VecDataset():

    def __init__(self, force_recompile=False):

        self.csv_filepath = WORD2VEC_DATASET_PATH

        #super().__init__(csv_filepath=WORD2VEC_DATASET_PATH)

        self.force_recompile = force_recompile

        #self.title = f"Word2Vec Embeddings"

        #breakpoint()
        #self.feature_cols = "TODO:" # feature_colnames(self.reducer_name, self.n_components)


    @cached_property
    def df(self):
        """Override parent method, compile dataset from reduction results."""
        if os.path.isfile(self.csv_filepath) and not self.force_recompile:
            print("LOADING EXISTING DATASET FROM FILE...")
            return read_csv(self.csv_filepath)
        else:
            print("COMPILING DATASET FROM RESULTS FILES...")
            ds = Dataset()
            labels_df = ds.labels #[colname for colname in  df.columns if not colname.isnumeric()]
            embeddings_df = read_csv(WORD2VEC_EMBEDDINGS_CSV_FILEPATH)
            df = labels_df.merge(embeddings_df, left_on="user_id", right_on="user_id")

            # write dataset (for faster loading later):
            df.to_csv(self.csv_filepath, index=False)
            return df


    @cached_property
    def x(self):
        """Override parent method, use feature cols specified below."""
        return self.df[self.feature_cols].copy()

    @property
    def feature_cols(self):
        """Features 0 through 99 (word2vec embeddings) """
        return [colname for colname in  self.df.columns if colname.isnumeric()]


    #@property
    #def label_cols(self):
    #    return [colname for colname in  self.df.columns if not colname.isnumeric()]



if __name__ == "__main__":



    ds = Word2VecDataset()

    print(ds.df.head())
