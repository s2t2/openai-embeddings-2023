
import os
import shutil
from functools import cached_property
from pprint import pprint

#from datetime import datetime
from itertools import chain
from collections import Counter

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess as tokenizer
from pandas import DataFrame, Series
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_STOPWORDS

from app import RESULTS_DIRPATH
from app.classification import Y_COLS

WORD2VEC_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "word2vec_embeddings")
#WORD2VEC_DESTRUCTIVE = bool(os.getenv("WORD2VEC_DESTRUCTIVE", default="false") == 'true')

#VECTOR_LENGTH = 100


class WordPipe:
    def __init__(self, corpus, tokenizer=tokenizer, results_dirpath=WORD2VEC_RESULTS_DIRPATH, stopwords=SKLEARN_STOPWORDS): # destructive=WORD2VEC_DESTRUCTIVE
        """Param corpus a pandas series of texts (text for each document)"""

        self.corpus = corpus
        self.tokenizer = tokenizer
        self.stopwords = stopwords

        #self.destructive = bool(destructive)
        self.results_dirpath = results_dirpath
        self.model_filepath = os.path.join(self.results_dirpath, f"w2v.model")
        #self.kv_filepath = os.path.join(self.results_dirpath, f"w2v.kv")
        self.word_vectors_csv_filepath = os.path.join(self.results_dirpath, "word_vectors.csv")
        self.document_vectors_csv_filepath = os.path.join(self.results_dirpath, "document_vectors.csv")


    @cached_property
    def corpus_tokens(self):
        return self.corpus.apply(tokenizer)

    @cached_property
    def word_counts(self):
        all_words = list(chain.from_iterable(self.corpus_tokens)) # h/t chat gpt for this one
        word_counter = Counter(all_words)
        return Series(word_counter.values(), index=word_counter.keys(), name="word_count")


    def perform(self):
        # TOKEN ANALYSIS (SIDE qUEST)
        print(self.word_counts.sort_values(ascending=False).head())

        self.load_or_train_model()
        print("WORDS:", len(self.words))

        print("WORD VECTORS:", self.word_vectors_df.shape) # 100 columns, default vector_size=100
        self.save_word_vectors()

        print("DOCUMENT VECTORS:", self.document_vectors.shape)
        self.save_document_vectors()


    def load_or_train_model(self, vector_size=100, window=10, min_count=2, workers=4):
        #if self.destructive:
        #    print("----------------")
        #    print("DESTRUCTIVE MODE...")
        #    #shutil.rmtree(self.results_dirpath)
        #    os.removedirs()

        os.makedirs(self.results_dirpath, exist_ok=True)

        if os.path.exists(self.model_filepath):
            print("----------------")
            print("LOADING MODEL FROM FILE...")
            print(self.model_filepath)
            self.model = Word2Vec.load(self.model_filepath)
            print(self.model)
            #print(type(self.model))
        else:
            print("----------------")
            print("INITIALIZING NEW MODEL...")
            self.model = Word2Vec(window=window, min_count=min_count, workers=workers, vector_size=vector_size)
            print(self.model)

            print("----------------")
            print("VOCAB...")
            self.model.build_vocab(self.corpus_tokens) # progress_per=1000
            #print("N SAMPLES:", model.corpus_count)
            #print("EPOCHS:", model.epochs)

            print("----------------")
            print("TRAINING...")
            self.model.train(self.corpus_tokens, total_examples=self.model.corpus_count, epochs=self.model.epochs)
            print(round(self.model.total_train_time, 0), "seconds")

            print("----------------")
            print("SAVING...")
            self.model.save(self.model_filepath)
            #self.model.wv.save(self.vectors_filepath)

        return self.model

    # AVAILABLE AFTER TRAINING:

    # WORD ANaLYSIS

    @property
    def words(self):
        return self.model.wv.index_to_key

    @property
    def word_vectors(self):
        return self.model.wv.vectors

    @property
    def word_vectors_df(self):
        return DataFrame(self.word_vectors, index=self.words)

    @cached_property
    def words_df(self):
        words_df = self.word_vectors_df.merge(self.word_counts, how="inner", left_index=True, right_index=True)
        words_df["is_stopword"] = words_df.index.map(lambda token: token in self.stopwords)
        words_df.index.name = "token"
        return words_df

    def save_word_vectors(self):
        self.words_df.to_csv(self.word_vectors_csv_filepath, index=True)

    # DOCUMENT ANALYSIS

    def infer_document_vector(self, tokens):
        """"Gets average vector for each set of tokens."""
        # Filter tokens that are in the model's vocabulary
        tokens = [token for token in tokens if token in self.model.wv.key_to_index]
        if any(tokens):
            # Calculate the average vector for the tokens in the document
            doc_vector = np.mean([self.model.wv[token] for token in tokens], axis=0)
        else:
            # If none of the tokens are in the model's vocabulary, return a zero vector
            doc_vector = np.zeros(self.model.vector_size)
        return doc_vector

    @cached_property
    def document_vectors(self):
        return self.corpus_tokens.apply(self.infer_document_vector)

    @cached_property
    def document_vectors_df(self, index_name="user_id"):
        # UNpacK EMBEdDINGS tO THEIR OWN COLUMNS
        docs_df = DataFrame(self.document_vectors.values.tolist())
        docs_df.columns = [str(i) for i in range(0, len(docs_df.columns))]
        docs_df.index = self.corpus_tokens.index
        docs_df.index.name = index_name
        return docs_df

    def save_document_vectors(self):
        self.document_vectors_df.to_csv(self.document_vectors_csv_filepath, index=True)


if __name__ == "__main__":


    from app.dataset import Dataset

    ds = Dataset()
    df = ds.df
    df.index = df["user_id"]

    #df["tokens"] = df["tweet_texts"].apply(tokenizer)
    #print(df["tokens"].head())

    wp = WordPipe(corpus=df["tweet_texts"])
    wp.perform()

    # INVEstIGatION
    # https://radimrehurek.com/gensim/models/keyedvectors.html
    wv = wp.model.wv #> gensim.models.keyedvectors.KeyedVectors
    print(len(wv))  #> 34,729 ORIGINAL ( ______ STOPwORD-REMOVED)

    #breakpoint()
    trumplike = wv.most_similar("realdonaldtrump", topn=10)
    pprint(trumplike)

    #wv.similarity(w1="impeachment", w2="sham")
    #wv.similarity(w1="impeachment", w2="just"))
    #wv.similarity(w1="impeachment", w2="fair"))
    #wv.similarity(w1="impeachment", w2="unfair"))
    #wv.similarity(w1="realdonaldtrump", w2="guilty"))
    #wv.similarity(w1="realdonaldtrump", w2="innocent"))
