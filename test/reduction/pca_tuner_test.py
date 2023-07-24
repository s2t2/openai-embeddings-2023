



from app.reduction.pca_tuner import PCATuner
from conftest import N_FEATURES

def test_pca_tuner(ds):

    tuner = PCATuner(ds=ds)

    assert "perform" in dir(tuner)
    assert "plot_explained_variance" in dir(tuner)
    assert "plot_scree" in dir(tuner)


def test_pca_tuner_performance(ds):

    max_components = 10

    tuner = PCATuner(ds=ds, max_components=max_components)
    assert tuner.max_components == max_components
    assert len(tuner.feature_names) == N_FEATURES

    tuner.perform()

    assert tuner.results_df.columns.tolist() == ["n_components", "explained_variance", "eigenvals"]
    assert tuner.results_df["n_components"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert tuner.results_df["explained_variance"].min() > 0
    assert tuner.results_df["explained_variance"].max() <= 1

#
