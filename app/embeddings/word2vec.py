
import os
import shutil

#from datetime import datetime
from itertools import chain
from collections import Counter

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess as tokenizer
from pandas import DataFrame, Series

from app import RESULTS_DIRPATH


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
    word_counts = Series(word_counter.values(), index=word_counter.keys())
    print(word_counts.sort_values(ascending=False).head())
    word_counts.to_json(os.path.join(WORD2VEC_RESULTS_DIRPATH, 'word_counts.json'))



    # MODEL TRAINING

    model = load_or_train_model(corpus=df["tokens"])
    print(type(model))

    # https://radimrehurek.com/gensim/models/keyedvectors.html
    wv = model.wv
    print(type(wv))

    #len(wv) #> 34,729

    # wv.index_to_key[0] #> "rt"

    #wv.most_similar("realdonaldtrump", topn=10)

    #wv.similarity(w1="impeachment", w2="sham")
    #wv.similarity(w1="impeachment", w2="just")
    #wv.similarity(w1="impeachment", w2="fair")
    #wv.similarity(w1="impeachment", w2="unfair")
    #wv.similarity(w1="impeachment", w2="witchhunt")
    #wv.similarity(w1="trump", w2="guilty")
    #wv.similarity(w1="trump", w2="innocent")

    vocab = wv.index_to_key
    print("VOCAB:", len(vocab))

    vectors = wv.vectors
    print("VECTORS:", vectors.shape) # why 100? oh it is a hyperparam. default vector_size=100

    vectors_df = DataFrame(wv.vectors, index=vocab)
    #print(vectors_df.shape)
    print(vectors_df.head())

    vectors_csv_filepath = os.path.join(WORD2VEC_RESULTS_DIRPATH, "word_vectors.csv")
    vectors_df.to_csv(vectors_csv_filepath)

    #pca_pipeline(x=vectors_df, chart_title="Word Embeddings")
