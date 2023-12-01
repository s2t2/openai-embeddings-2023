
import os
#import shutil
from functools import cached_property
from pprint import pprint
import joblib
#from datetime import datetime
from itertools import chain
from collections import Counter
import re

import numpy as np
from pandas import DataFrame, Series
from gensim.utils import simple_preprocess as tokenizer # lowercases, tokenizes
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS as SKLEARN_STOPWORDS


from app import RESULTS_DIRPATH, save_results_json
from app.nlp import convert_non_ascii
from app.classification import Y_COLS

TFIDF_RESULTS_DIRPATH = os.path.join(RESULTS_DIRPATH, "tfidf_embeddings")


#def remove_non_ascii(my_text):
#    return my_text.encode('ascii', 'ignore').decode('ascii')


class TextEmbeddingPipeline:
    def __init__(self, corpus, tokenizer=tokenizer, stopwords=list(SKLEARN_STOPWORDS), results_dirpath=TFIDF_RESULTS_DIRPATH): # save_results=True, # destructive=WORD2VEC_DESTRUCTIVE
        """Param corpus a pandas series of texts (text for each document)"""

        self.corpus = corpus
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        #self.save_results = bool(save_results)
        #self.destructive = bool(destructive)

        self.corpus = self.corpus.apply(convert_non_ascii) # 72_854

        self.results_dirpath = results_dirpath
        self.model_filepath = os.path.join(self.results_dirpath, f"tfidf.model")
        self.results_json_filepath = os.path.join(self.results_dirpath, "results.json")
        self.terms_csv_filepath = os.path.join(self.results_dirpath, "terms.csv")
        self.document_embeddings_csv_filepath = os.path.join(self.results_dirpath, "documents.csv.gz")
        self.document_embeddings_hd5_filepath = os.path.join(self.results_dirpath, "documents.hd5")

        # after training;
        self.model = None
        self.embeddings = None
        self.vocab = None
        self.feature_names = None
        self.embbedings_df = None
        self.results = None


    @cached_property
    def corpus_tokens_in(self):
        """without stopword removal"""
        return self.corpus.apply(tokenizer)

    @cached_property
    def word_counts_in(self):
        """without stopword removal"""
        all_words = list(chain.from_iterable(self.corpus_tokens_in)) # h/t chat gpt for this one
        word_counter = Counter(all_words)
        return Series(word_counter.values(), index=word_counter.keys(), name="word_count")

    @cached_property
    def words_df(self):
        """before stopword removal, etc"""
        words_df = DataFrame(self.word_counts_in)
        words_df["is_stopword"] = words_df.index.map(lambda token: token in self.stopwords)
        words_df.index.name = "token"
        return words_df

    @cached_property
    def top_words_df(self):
        top_n=1_000
        return self.words_df.sort_values(by="word_count", ascending=False).head(top_n)

    def perform(self):
        # TOKEN ANALYSIS (SIDE qUEST)
        print("----------------")
        print("WORDS IN:", len(self.word_counts_in)) #> 72_026 terms in
        print(self.top_words_df.head())

        print("----------------")
        print(f"STOPWORDS ({len(self.stopwords)}):")
        print(sorted(list(self.stopwords)))

        print("----------------")
        print("INITIALIZING NEW MODEL...")

        self.model = TfidfVectorizer(tokenizer=self.tokenizer, stop_words=self.stopwords)
        print(self.model)

        print("----------------")
        print("TRAINinG...")

        self.embeddings = self.model.fit_transform(self.corpus)
        print("DOCUMENT EMBEDdINGS:", self.embeddings.shape) #> (7_566, 72_854)

        # mapping of terms to feature indices:
        self.vocab = self.model.vocabulary_ #> {"rt": 53304, "foxnewpolls": 21286, "poll": 46945, "donald": 15855}
        # terms, in alphabetical order
        self.feature_names = self.model.get_feature_names_out()
        #print("FEATURE NAMES:", len(self.feature_names))
        assert self.vocab[self.feature_names[0]] == 0

        self.embeddings_df = DataFrame(self.embeddings.toarray(), index=self.corpus.index, columns=self.feature_names)
        print(self.embeddings_df.head())

        print("----------------")
        print("SAVING...")

        params = self.model.get_params()
        params["dtype"] = str(params["dtype"]) # serializable class name
        params["ngram_range"] = list(params["ngram_range"]) # serializable tuple
        params["tokenizer"] = str(params["tokenizer"].__name__) # serializable function name
        params["stop_words"] = sorted(params["stop_words"])
        self.results = {
            "model": str(self.model.__class__.__name__),
            "params": params,
            "vocab": len(self.feature_names),
            "embeddings": self.embeddings_df.shape
        }

        #print("...RESUltS...")
        save_results_json(self.results, self.results_json_filepath)

        #print("...wORD FREQUENCIES...")
        self.top_words_df.to_csv(self.terms_csv_filepath, index=True)
        #print("...DOCUMENT EMBeDDINGS...")
        #self.embeddings_df.to_csv(self.document_embeddings_csv_filepath, index=True) # TAKeS TOO LONG? tOO SPArSE? tOO MANY COlS?
        self.embeddings_df.to_hdf(self.document_embeddings_hd5_filepath, index=True, key="document_embeddings")

        #print("... MODEL...")
        #joblib.dump(self.model, self.model_filepath)



if __name__ == "__main__":


    from app.dataset import Dataset

    ds = Dataset()
    df = ds.df
    df.index = df["user_id"]

    pipeline = TextEmbeddingPipeline(corpus=df["tweet_texts"])
    pipeline.perform()
