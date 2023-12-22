import os
from pandas import DataFrame, read_csv, read_hdf
import numpy as np


from app.reduction.pipeline import ReductionPipeline, REDUCER_TYPE, N_COMPONENTS
from app.tfidf_embeddings.pipeline import TFIDF_RESULTS_DIRPATH


DOCUMENT_EMBEDDINGS_HD5_FILEPATH = os.path.join(TFIDF_RESULTS_DIRPATH, "documents.hd5")


class TFIDFReductionPipeline(ReductionPipeline):

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




if __name__ == "__main__":

    from app.dataset import Dataset
    from app.colors import COLORS_MAP, CATEGORY_ORDERS

    ds = Dataset()
    df = ds.df

    print("------------")
    print("DOCUMENT EMBEDDINGS...")

    #pipeline = TextEmbeddingPipeline(corpus=df["tweet_texts"])
    #pipeline.perform()
    #embeddings_df = pipeline.embeddings_df
    #embeddings_df = read_csv(pipeline.document_embeddings_csv_filepath)
    embeddings_df = read_hdf(DOCUMENT_EMBEDDINGS_HD5_FILEPATH, key="document_embeddings")
    print(embeddings_df.shape)
    # TypeError: Feature names are only supported if all input features have string names,
    # but your input has ['float', 'str'] as feature name / column name types.
    # If you want feature names to be stored and validated, you must convert them all to strings,
    # by using X.columns = X.columns.astype(str) for example.
    # Otherwise you can remove feature / column names from your input data, or convert them all to a non-string data type.
    embeddings_df.columns = embeddings_df.columns.astype(str)

    doc_results_dirpath = os.path.join(TFIDF_RESULTS_DIRPATH, "doc_reduction")
    doc_labels_df = ds.labels.copy()
    doc_labels_df.index = doc_labels_df["user_id"] # for joining the datasets together later

    for reducer_type in ["PCA", "T-SNE", "UMAP"]:
        print(reducer_type)

        drp = TFIDFReductionPipeline(x=embeddings_df, labels_df=doc_labels_df, reducer_type=reducer_type, n_components=2, results_dirpath=doc_results_dirpath)
        drp.perform()
        drp.save_embeddings()

        subtitle = "TF-IDF Document Embeddings (User Tweet Timelines)"
        drp.plot_embeddings(subtitle=subtitle)

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
