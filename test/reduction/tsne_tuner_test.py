



from app.reduction.tsne_tuner import TSNETuner
from conftest import N_FEATURES

def test_tsne_tuner(ds):

    tuner = TSNETuner(ds=ds)

    assert "perform" in dir(tuner)
    assert "plot_kl_divergence" in dir(tuner)


def test_tsne_tuner_performance(ds):

    max_components = 3 # tsne gets slow after 4!

    tuner = TSNETuner(ds=ds, max_components=max_components)
    assert tuner.max_components == max_components
    assert len(tuner.feature_names) == N_FEATURES

    tuner.perform()
    assert tuner.results_df.columns.tolist() == ["n_components", "kl_divergence"]
    assert tuner.results_df["n_components"].tolist() == [1, 2, 3]
    kl =  tuner.results_df["kl_divergence"].tolist()
    assert min(kl) > 0
    assert kl == sorted(kl, reverse=True)
