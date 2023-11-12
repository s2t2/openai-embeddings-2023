import os

from app.reduction.pipeline import ReductionPipeline, REDUCER_TYPE, N_COMPONENTS
from app.embeddings.word2vec import WORD2VEC_RESULTS_DIRPATH, WordPipe


class WordVectorReductionPipeline(ReductionPipeline):

    def __init__(self, x, results_dirpath,
                 reducer_type=REDUCER_TYPE, n_components=N_COMPONENTS,
                 labels_df=None, #x_scale=X_SCALE,
                ):

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
        results_df.index.name = "token"
        #results_df["token"] = self.x.index

        for colname in self.component_names:
            # rename column to include info about which method produced it:
            results_df.rename(columns={colname: f"{self.reducer_name}_{self.n_components}_{colname}"}, inplace=True)
        results_df.to_csv(csv_filepath, index=True)





if __name__ == "__main__":


    wp = WordPipe()

    # DIMENSIONALITY REducTION ON WORD VECTORS
    reduce_words = True
    if reduce_words:
        #pca_pipeline(x=vectors_df, chart_title="Word Embeddings")

        for reducer_type in ["PCA", "T-SNE", "UMAP"]:

            breakpoint()
            word_results_filepath = os.path.join(WORD2VEC_RESULTS_DIRPATH, "tokens")
            drp = WordVectorReductionPipeline(x=vectors_df, labels_df=self.word_counts.to_frame(), reducer_type=reducer_type, n_components=2, results_dirpath=word_results_filepath)
            drp.perform()

            drp.embeddings_df = drp.embeddings_df.merge(word_counts, how="inner", left_index=True, right_index=True)
            drp.embeddings_df["token"] = drp.embeddings_df.index
            drp.save_embeddings()

            #TOP_N = 250
            #drp.embeddings_df.sort_values(by=["word_count"], ascending=False, inplace=True) # it is already sorted, but just to be sure
            #drp.embeddings_df = drp.embeddings_df.head(TOP_N)
            # oh this is not that interesting unless we perform stopword removal
            #drp.plot_embeddings(size="word_count", hover_data=["token", "word_count"]) # subtitle=f"Top {TOP_N} Words"
            drp.plot_embeddings(hover_data=["token", "word_count"], subtitle="Word2Vec Embeddings of Word Vectors")


    # dIMensionaLITy REductIon ON DOcUMENTS (USER TIMELINE TWEETS)
    reduce_docs = True
    if reduce_docs:
        for reducer_type in ["PCA", "T-SNE", "UMAP"]:

            breakpoint()
            doc_results_dirpath = os.path.join(WORD2VEC_RESULTS_DIRPATH, "documents")
            drp = WordVectorReductionPipeline(x=embeds_df, reducer_type=reducer_type, n_components=2, results_dirpath=doc_results_dirpath)
            drp.perform()

            breakpoint()

            drp.embeddings_df = drp.embeddings_df.merge(word_counts, how="inner", left_index=True, right_index=True)
            #drp.embeddings_df["token"] = drp.embeddings_df.index

            drp.save_embeddings()

            drp.plot_embeddings(subtitle="Word2Vec Embeddings of User Tweet Timelines")

            #drp.plot_embeddings(subtitle="Word2Vec Embeddings of User Tweet Timelines", color="bot_label")
            #drp.plot_embeddings(subtitle="Word2Vec Embeddings of User Tweet Timelines", color="opinion_community")
            #drp.plot_embeddings(subtitle="Word2Vec Embeddings of User Tweet Timelines", color="toxic_label")
            #drp.plot_embeddings(subtitle="Word2Vec Embeddings of User Tweet Timelines", color="fact_label")
            #drp.plot_embeddings(subtitle="Word2Vec Embeddings of User Tweet Timelines", color="fourway_label")
