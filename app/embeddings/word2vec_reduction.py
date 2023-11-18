import os
from pandas import DataFrame
import numpy as np


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
        os.makedirs(self.results_dirpath, exist_ok=True)

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

    from app.dataset import Dataset
    from app.colors import COLORS_MAP, CATEGORY_ORDERS

    ds = Dataset()
    df = ds.df

    wp = WordPipe(corpus=df["tweet_texts"])
    wp.load_or_train_model()

    print("------------")
    print("WORD EMBEDDINGS...")

    word_results_filepath = os.path.join(wp.results_dirpath, "word_reduction")
    word_labels_df = wp.words_df[["word_count"]]
    for reducer_type in ["PCA", "T-SNE", "UMAP"]: #
        print(reducer_type)

        drp = WordVectorReductionPipeline(x=wp.word_vectors_df, labels_df=word_labels_df, reducer_type=reducer_type, n_components=2, results_dirpath=word_results_filepath)
        drp.perform()
        drp.save_embeddings()

        drp.embeddings_df = drp.embeddings_df.merge(wp.words_df["is_stopword"], how="inner", left_index=True, right_index=True)
        drp.embeddings_df["token"] = drp.embeddings_df.index
        drp.plot_embeddings(hover_data=["token", "word_count"], subtitle="Word2Vec Word Embeddings", color="is_stopword")

        # special chart
        #drp.embeddings_df = drp.embeddings_df[drp.embeddings_df["is_stopword"] == False]
        ##drp.plot_embeddings(hover_data=["token", "word_count"], subtitle="Word2Vec Word Embeddings (excluding stopwords)", size="word_count")
        ## oh this is not that interesting unless we perform stopword removal
        #TOP_N = 50
        #drp.embeddings_df['scaled_count'] = np.log10(drp.embeddings_df['word_count']) # special special
        #
        #drp.embeddings_df.sort_values(by=["word_count"], ascending=False, inplace=True) # it is already sorted, but just to be sure
        #drp.embeddings_df = drp.embeddings_df.head(TOP_N)
        #drp.plot_embeddings(size="scaled_count",
        #                    color='scaled_count', color_scale="oranges",
        #                    hover_data=["token", "word_count"],
        #                    subtitle=f"Top {TOP_N} Words",
        #                    text="token",
        #                    )

        #exit()

    exit()

    print("------------")
    print("DOCUMENT EMBEDDINGS...")

    doc_results_dirpath = os.path.join(wp.results_dirpath, "doc_reduction")
    doc_labels_df = ds.labels.copy()
    for reducer_type in ["PCA", "T-SNE", "UMAP"]:
        print(reducer_type)

        drp = WordVectorReductionPipeline(x=wp.document_vectors_df, labels_df=doc_labels_df, reducer_type=reducer_type, n_components=2, results_dirpath=doc_results_dirpath)
        drp.perform()
        drp.save_embeddings()

        subtitle = "Word2Vec Document Embeddings (User Tweet Timelines)"
        drp.plot_embeddings(subtitle=subtitle)
        #drp.plot_embeddings(subtitle=subtitle, color="bot_label")
        #drp.plot_embeddings(subtitle=subtitle, color="opinion_community")
        #drp.plot_embeddings(subtitle=subtitle, color="toxic_label")
        #drp.plot_embeddings(subtitle=subtitle, color="fact_label")
        #drp.plot_embeddings(subtitle=subtitle, color="fourway_label")

        for groupby_col in [
            "bot_label", "opinion_label", "bom_overall_label", "bom_astroturf_label",
            "toxic_label", "factual_label",
            "fourway_label", #"sixway_label",
                        ]:
            color_map = COLORS_MAP[groupby_col]
            category_orders = {groupby_col: CATEGORY_ORDERS[groupby_col]}

            results_dirpath = os.path.join(doc_results_dirpath, groupby_col)
            os.makedirs(results_dirpath, exist_ok=True)

            drp.plot_embeddings(color=groupby_col, color_map=color_map,
                                        category_orders=category_orders,
                                    #hover_data=["user_id", "bot_label"],
                                    #fig_show=True, fig_save=True,
                                    results_dirpath=results_dirpath, subtitle=subtitle
                                    )

            drp.plot_centroids(groupby_col=groupby_col, color_map=color_map,
                                    category_orders=category_orders,
                                #hover_data=["user_id", "bot_label"],
                                #fig_show=True, fig_save=True
                                results_dirpath=results_dirpath, subtitle=subtitle
                                )
