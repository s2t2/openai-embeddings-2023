

#import numpy as np
#from pandas import read_csv
#from pytest import fixture


from app.reduction_pipeline import ReductionPipeline
from conftest import N_USERS


def test_pca_pipeline(ds):

    pipeline = ReductionPipeline(df=ds.df, label_cols=ds.label_cols,
                                 reducer_type="PCA", n_components=2)
    pipeline.perform()

    # just the embeddings resulting from PCA
    embeddings = pipeline.embeddings
    assert embeddings.shape == (N_USERS, 2)

    # embeddings resulting from PCA, plus label columns (for easier analysis and charting later)
    embeddings_df = pipeline.embeddings_df
    assert embeddings_df.shape == (7566, 25)
    assert embeddings_df.columns.tolist() == [
        'component_1', 'component_2',
        'user_id', 'created_on', 'screen_name_count', 'screen_names', 'status_count', 'rt_count',
        'rt_pct', 'avg_toxicity', 'avg_fact_score',
        'opinion_community', 'is_bot', 'is_q',
        'tweet_texts',
        'bom_cap', 'bom_astroturf','bom_fake_follower', 'bom_financial', 'bom_other',
        'opinion_label', 'bot_label', 'q_label', 'fourway_label', 'sixway_label'
    ]








#def test_tsne_pipeline(features_df):
#    pipeline = ReductionPipeline(features_df, reducer_type="T-SNE", n_components=2)
#    pipeline.perform()
#    verify_embeddings(pipeline)
#
#
#def test_umap_pipeline(features_df):
#    pipeline = ReductionPipeline(features_df, reducer_type="UMAP", n_components=2)
#    pipeline.perform()
#    verify_embeddings(pipeline)
#

#def test_pca_explainability(audio_features_df):
#    feature_names = ['tempo', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var', 'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'spectral_rolloff_mean', 'spectral_rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var', 'tonnetz_mean', 'tonnetz_var', 'mfcc_1_mean', 'mfcc_1_var', 'mfcc_2_mean', 'mfcc_2_var', 'mfcc_3_mean', 'mfcc_3_var', 'mfcc_4_mean', 'mfcc_4_var', 'mfcc_5_mean', 'mfcc_5_var', 'mfcc_6_mean', 'mfcc_6_var', 'mfcc_7_mean', 'mfcc_7_var', 'mfcc_8_mean', 'mfcc_8_var', 'mfcc_9_mean', 'mfcc_9_var', 'mfcc_10_mean', 'mfcc_10_var', 'mfcc_11_mean', 'mfcc_11_var', 'mfcc_12_mean', 'mfcc_12_var', 'mfcc_13_mean', 'mfcc_13_var']
#
#    pipeline = ReductionPipeline(audio_features_df, reducer_type="PCA", n_components=2)
#    pipeline.perform()
#    verify_embeddings(pipeline)
#
#    pca = pipeline.reducer
#    assert np.allclose(pca.explained_variance_.tolist(), [8.980046729035566, 7.014918995282904])
#    assert np.allclose(pca.explained_variance_ratio_.tolist(), [0.21890505388738124, 0.17100147326771628])
#    assert np.allclose(pca.singular_values_.tolist(), [127.73701463028492, 112.89866170344553])
#    assert pca.feature_names_in_.tolist() == feature_names
#
#    loadings = pipeline.loadings
#    assert loadings.shape == (41, 2)
#    assert loadings.min() > -1
#    assert loadings.max() < 1
#    loadings_df = pipeline.loadings_df
#    assert loadings_df.columns.tolist() == ["component_1", "component_2"]
#    assert loadings_df.index.tolist() == feature_names
#
#    # these represent the absolute magnitude of importances, not direction up or down
#    feature_importances = pipeline.feature_importances
#
#    c1 = feature_importances["component_1"]
#    assert list(c1.keys()) == ['mfcc_8_var', 'mfcc_7_var', 'mfcc_6_var', 'mfcc_9_var', 'mfcc_10_var', 'mfcc_4_var', 'mfcc_5_var', 'spectral_centroid_var', 'mfcc_11_var', 'mfcc_2_var']
#    #assert np.allclose(list(c1.values()), [0.805443354102087,0.7973620573318918,0.79285956726038,0.7860296338757697,0.7663541218866315,0.756703823949484,0.7230504963967018,0.7110909325450872,0.6849105101981539,0.6737980228824214])
#
#    c2 = feature_importances["component_2"]
#    assert list(c2.keys()) == ['spectral_bandwidth_mean','spectral_rolloff_mean','spectral_centroid_mean','mfcc_2_mean','chroma_stft_mean','mfcc_1_mean','tonnetz_var','mfcc_8_mean','mfcc_10_mean','zero_crossing_rate_mean']
#    #assert np.allclose(list(c2.values()), [0.8539164806642479, 0.8464210286829734, 0.8184160145817183, 0.8120671323088787, 0.743604778175382, 0.6886499046288507, 0.642432984513275, 0.5722250679251756, 0.5581507671324493, 0.5227770916583789])
#
#
#
#
#
#
