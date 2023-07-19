

import numpy as np

from app.reduction_pipeline import ReductionPipeline
from conftest import N_USERS, N_FEATURES


def verify_embeddings(pipeline):
    """A helper method for testing the embeddings for a given pipeline. Assumes two components were used."""

    n_components = pipeline.n_components
    component_names = pipeline.component_names
    label_cols = [
        'user_id', 'created_on', 'screen_name_count', 'screen_names',
        'status_count', 'rt_count', 'rt_pct', 'avg_toxicity', 'avg_fact_score',
        'opinion_community', 'is_bot', 'is_q',
        'tweet_texts',
        'bom_cap', 'bom_astroturf','bom_fake_follower', 'bom_financial', 'bom_other',
        'opinion_label', 'bot_label', 'q_label', 'fourway_label', 'sixway_label'
    ]
    expected_cols = component_names + label_cols # joining these together dynamically to allow us to test different number of components / avoid hard coding 1 and 2 only
    n_cols = len(expected_cols)

    # embeddings resulting from dimensionality reduction (no labels)
    embeddings = pipeline.embeddings
    assert embeddings.shape == (N_USERS, n_components)

    # embeddings resulting from dimensionality reduction (plus labels)
    embeddings_df = pipeline.embeddings_df
    assert embeddings_df.shape == (N_USERS, n_cols)
    assert embeddings_df.columns.tolist() == expected_cols

def verify_pca_explainability(pipeline, dataset):
    """A helper method for testing the explainability metrics returned by PCA specifically."""

    n_components = pipeline.n_components # 2
    component_names = pipeline.component_names # ["component_1", "component_2"]
    pca = pipeline.reducer

    explained_var = pca.explained_variance_.tolist()
    assert len(explained_var) == n_components

    explained_var_ratios = pca.explained_variance_ratio_.tolist()
    assert len(explained_var_ratios) == n_components
    # the values are between 0 and 1, sorted in descending order
    assert explained_var_ratios == sorted(explained_var_ratios, reverse=True)
    assert min(explained_var_ratios) > 0 and max(explained_var_ratios) <= 1

    singular_vals = pca.singular_values_.tolist()
    assert len(singular_vals) == n_components

    feature_names = pca.feature_names_in_.tolist()
    assert len(feature_names) == N_FEATURES
    assert feature_names == dataset.feature_names

    loadings = pipeline.loadings
    assert loadings.shape == (N_FEATURES, n_components)
    assert loadings.min() > -1 and loadings.max() < 1

    loadings_df = pipeline.loadings_df
    assert loadings_df.columns.tolist() == component_names
    assert loadings_df.index.tolist() == feature_names

    feature_importances = pipeline.feature_importances
    # returns the top ten features and their importances, for each component
    # these represent the absolute magnitude of importances, not direction up or down
    #> {
    #>     'component_1': {'26': 0.684083389074209,'267': 0.6889728933226322,'286': 0.702036938280094,'361': 0.7620002964420207,'397': 0.7290441887904475,'484': 0.7225299547443916,'519': 0.781550378345577,'640': 0.6720619483492493,'657': 0.6835787634568226,'970': 0.6861974623404744},
    #>     'component_2': {'1015': 0.44419697955251625,'1104': 0.44413654720670565,'1311': 0.42051099405700815,'194': 0.4899146795868501,'197': 0.4179642454260487,'228': 0.484664720736489,'426': 0.4181673919111419,'5': 0.4161947777000336,'578': 0.428198251700284,'711': 0.41623196143027624}
    #> }
    assert list(feature_importances.keys()) == component_names
    for _, important_features in feature_importances.items():
        important_feature_names = important_features.keys()
        assert len(important_feature_names) == 10
        importances = important_features.values()
        assert len(importances) == 10
        assert min(importances) > 0 and max(importances) < 1

def verify_tsne_explainability(pipeline):
    kl_divergence = pipeline.reducer.kl_divergence_
    assert kl_divergence >= 0


def test_pca_pipeline(ds):
    pipeline = ReductionPipeline(df=ds.df, label_cols=ds.label_cols, reducer_type="PCA", n_components=2)
    pipeline.perform()
    verify_embeddings(pipeline)
    verify_pca_explainability(pipeline, ds)

def test_tsne_pipeline(ds):
    pipeline = ReductionPipeline(df=ds.df, label_cols=ds.label_cols, reducer_type="T-SNE", n_components=2)
    pipeline.perform()
    verify_embeddings(pipeline)
    verify_tsne_explainability(pipeline)

def test_umap_pipeline(ds):
    pipeline = ReductionPipeline(df=ds.df, label_cols=ds.label_cols, reducer_type="UMAP", n_components=2)
    pipeline.perform()
    verify_embeddings(pipeline)
