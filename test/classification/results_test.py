
from app.classification.logistic_regression import LogisticRegressionPipeline
from app.classification.random_forest import RandomForestPipeline
from app.classification.xgboost import XGBoostPipeline


logistic_params_grid = {"classifier__max_iter": [25]}
forest_params_grid = {"classifier__criterion": ["gini"]}


def results_ok(pipeline):
    """Shared assertions about the results"""
    results = pipeline.results
    assert isinstance(results.accy, float)
    assert isinstance(results.f1_macro, float)
    assert isinstance(results.f1_weighted, float)
    assert isinstance(results.roc_auc_score, float)
    #assert isinstance(results.roc_auc_score_proba, float)
    #assert results.roc_auc_score_proba == results.roc_curve_auc

    results_json = pipeline.results_json
    assert sorted(results_json.keys()) == [
        'class_labels', 'class_names',
        'classification_report', 'confusion_matrix', 'grid_search',
        'roc_auc_score', 'x_scaled', 'y_col'
    ]
    assert isinstance(results_json["roc_auc_score"], float)
    assert sorted(results_json["grid_search"].keys()) == [
        'best_params', 'best_score', 'k_folds', 'model_type', 'param_grid',
    ]
    #assert list(results_json["classification_report"].keys()) == expected_class_names + ['accuracy', 'macro avg', 'weighted avg']
    for metric in ['accuracy', 'macro avg', 'weighted avg']:
        assert metric in results_json["classification_report"].keys()

    # PREDICTIONS REPORT

    preds_df = pipeline.predictions_df #results.predictions_df
    assert preds_df.shape == (len(results.y_test), 9)
    assert preds_df.columns.tolist() == [
        "user_id", "is_bot", "opinion_community", "is_toxic", "is_factual", "fourway_label", "tweet_texts",
        "y_test", "y_pred"
    ]


def plots_ok(pipeline):
    pipeline.plot_confusion_matrix(fig_save=False, fig_show=False)

    if pipeline.is_multiclass:
        pipeline.plot_roc_curve_multiclass(fig_save=False, fig_show=False)
    else:
        pipeline.plot_roc_curve(fig_save=False, fig_show=False)



def test_logistic_regression_binary(ds):
    pipeline = LogisticRegressionPipeline(ds=ds, y_col="is_bot", param_grid=logistic_params_grid)
    assert pipeline.n_classes == 2
    pipeline.train_eval()
    results_ok(pipeline)
    plots_ok(pipeline)


def test_random_forest_binary(ds):
    pipeline = RandomForestPipeline(ds=ds, y_col="is_bot", param_grid=forest_params_grid)
    assert pipeline.n_classes == 2
    pipeline.train_eval()
    results_ok(pipeline)
    plots_ok(pipeline)

def test_xgboost_binary(ds):
    pipeline = XGBoostPipeline(ds=ds, y_col="is_bot", param_grid={})
    assert pipeline.n_classes == 2
    pipeline.train_eval()
    results_ok(pipeline)
    plots_ok(pipeline)




def test_logistic_regression_multiclass(ds):
    pipeline = LogisticRegressionPipeline(ds=ds, y_col="fourway_label", param_grid=logistic_params_grid)
    assert pipeline.n_classes == 4
    pipeline.train_eval()
    results_ok(pipeline)
    plots_ok(pipeline)


def test_random_forest_multiclass(ds):
    pipeline = RandomForestPipeline(ds=ds, y_col="fourway_label", param_grid=forest_params_grid)
    assert pipeline.n_classes == 4
    pipeline.train_eval()
    results_ok(pipeline)
    plots_ok(pipeline)

def test_xgboost_multiclass(ds):
    pipeline = XGBoostPipeline(ds=ds, y_col="fourway_label", param_grid={})
    assert pipeline.n_classes == 4
    pipeline.train_eval()
    results_ok(pipeline)
    plots_ok(pipeline)
