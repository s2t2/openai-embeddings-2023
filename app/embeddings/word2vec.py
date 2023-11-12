
import os
import shutil

#from datetime import datetime
from itertools import chain
from collections import Counter

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess as tokenizer
from pandas import DataFrame, Series

from app import RESULTS_DIRPATH
from app.reduction.pipeline import ReductionPipeline, REDUCER_TYPE, N_COMPONENTS

WORD2VEC_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "embeddings", "word2vec")
WORD2VEC_DESTRUCTIVE = bool(os.getenv("WORD2VEC_DESTRUCTIVE", default="false") == 'true')

#VECTOR_LENGTH = 100


def load_or_train_model(corpus, results_dirpath=WORD2VEC_RESULTS_DIRPATH,
                        vector_size=100, window=10, min_count=2, workers=4,
                        destructive=WORD2VEC_DESTRUCTIVE
                        ):
    """
    Params corpus (pandas.Series) : a column of tokenized text, can be variable length
    """

    if destructive:
        print("DESTRUCTIVE MODE...")
        shutil.rmtree(results_dirpath)

    os.makedirs(results_dirpath, exist_ok=True)

    model_filepath = os.path.join(results_dirpath, f"my-model.model")
    vectors_filepath = os.path.join(results_dirpath, f"my-model.kv")

    if os.path.exists(model_filepath):
        print("LOADING MODEL FROM FILE...")
        print(model_filepath)
        model = Word2Vec.load(model_filepath)
    else:

        print("INITIALIZING NEW MODEL...")
        model = Word2Vec(window=window, min_count=min_count, workers=workers, vector_size=vector_size)

        print("VOCAB...")
        model.build_vocab(corpus) # progress_per=1000
        #print("N SAMPLES:", model.corpus_count)
        #print("EPOCHS:", model.epochs)

        print("TRAINING...")
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        print(round(model.total_train_time, 0), "seconds")

        print("SAVING...")
        model.save(model_filepath)
        model.wv.save(vectors_filepath)

    return model



class AnotherReductionPipeline(ReductionPipeline):

    def __init__(self, x, labels_df=None, #x_scale=X_SCALE,
                        reducer_type=REDUCER_TYPE, n_components=N_COMPONENTS,
                        results_dirpath=WORD2VEC_RESULTS_DIRPATH):

        self.x = x
        self.labels_df = labels_df

        self.reducer_type = reducer_type
        self.n_components = n_components
        self.results_dirpath = results_dirpath


        self.reducer_name = {"PCA": "pca", "T-SNE": "tsne", "UMAP": "umap"}[self.reducer_type]

        self.reducer = None
        self.embeddings = None
        self.embeddings_df = None
        self.loadings = None
        self.loadings_df = None


    def save_embeddings(self):
        """
        Save a slim copy of the embeddings to CSV (just user_id and component values).
        With the goal of merging all the results into a single file later.
        """
        csv_filepath = os.path.join(self.results_dirpath, f"{self.reducer_name}_{self.n_components}_embeddings.csv")

        results_df = self.embeddings_df.copy()
        #results_df.index = self.x.index
        #results_df.index.name = "token"
        results_df["token"] = self.x.index

        for colname in self.component_names:
            # rename column to include info about which method produced it:
            results_df.rename(columns={colname: f"{self.reducer_name}_{self.n_components}_{colname}"}, inplace=True)
        results_df.to_csv(csv_filepath, index=False)



if __name__ == "__main__":


    from app.dataset import Dataset

    ds = Dataset()
    df = ds.df

    # TEXT PROCESSING / TOKENIZATION

    df["tokens"] = df["tweet_texts"].apply(tokenizer)
    print(df["tokens"].head())

    # TOKEN ANALYSIS (SIDE qUEST)

    all_words = list(chain.from_iterable(df["tokens"])) # h/t chat gpt for this one
    unique_words = list(set(all_words))
    print("NUMBER OF UNIQUE TOKENS:", len(unique_words))

    word_counter = Counter(all_words)
    # len(word_counter.keys()) #> unique words
    word_counts = Series(word_counter.values(), index=word_counter.keys(), name="word_count")
    print(word_counts.sort_values(ascending=False).head())
    #word_counts.to_json(os.path.join(WORD2VEC_RESULTS_DIRPATH, 'word_counts.json'))

    # MODEL TRAINING

    model = load_or_train_model(corpus=df["tokens"])
    print(type(model))

    # https://radimrehurek.com/gensim/models/keyedvectors.html
    wv = model.wv
    print(type(wv))
    #len(wv) #> 34,729
    # wv.index_to_key[0] #> "rt"

    vocab = wv.index_to_key
    print("WORDS:", len(vocab))

    vectors = wv.vectors
    print("WORD VECTORS:", vectors.shape) # 100 columns, default vector_size=100

    vectors_df = DataFrame(wv.vectors, index=vocab)
    #print(vectors_df.shape)
    print(vectors_df.head())

    vectors_csv_filepath = os.path.join(WORD2VEC_RESULTS_DIRPATH, "word_vectors.csv")
    vectors_df.to_csv(vectors_csv_filepath)

    #wv.most_similar("realdonaldtrump", topn=10)

    #wv.similarity(w1="impeachment", w2="sham")
    #wv.similarity(w1="impeachment", w2="just")
    #wv.similarity(w1="impeachment", w2="fair")
    #wv.similarity(w1="impeachment", w2="unfair")
    #wv.similarity(w1="impeachment", w2="witchhunt")
    #wv.similarity(w1="trump", w2="guilty")
    #wv.similarity(w1="trump", w2="innocent")

    #pca_pipeline(x=vectors_df, chart_title="Word Embeddings")

    drp = AnotherReductionPipeline(x=vectors_df, n_components=2, results_dirpath=WORD2VEC_RESULTS_DIRPATH)
    drp.perform()

    drp.embeddings_df = drp.embeddings_df.merge(word_counts, how="inner", left_index=True, right_index=True)
    drp.embeddings_df["token"] = drp.embeddings_df.index
    drp.save_embeddings()

    #TOP_N = 250
    #drp.embeddings_df.sort_values(by=["word_count"], ascending=False, inplace=True) # it is already sorted, but just to be sure
    #drp.embeddings_df = drp.embeddings_df.head(TOP_N)
    # oh this is not that interesting unless we perform stopword removal
    #drp.plot_embeddings(size="word_count", hover_data=["token", "word_count"]) # subtitle=f"Top {TOP_N} Words"
    drp.plot_embeddings(hover_data=["token", "word_count"])
