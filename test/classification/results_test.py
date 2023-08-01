
from app.classification.logistic_regression import MyLogisticRegression
from app.classification.random_forest import MyRandomForest

logistic_params_grid = {"classifier__max_iter": [25]}
forest_params_grid = {"classifier__criterion": ["gini"]}

def results_ok(pipeline):
    results = pipeline.results
    results_json = pipeline.results_json
    assert isinstance(results.accy, float)
    assert isinstance(results.f1_macro, float)
    assert isinstance(results.f1_weighted, float)
    assert isinstance(results.roc_auc_score, float)
    assert sorted(results_json.keys()) == ['class_names', 'classification_report', 'confusion_matrix', 'grid_search', 'roc_auc_score']
    assert isinstance(results_json["roc_auc_score"], float)
    assert sorted(results_json["grid_search"].keys()) == ['best_params', 'best_score', 'k_folds', 'model_type', 'param_grid', 'x_scaled']

    #assert list(results_json["classification_report"].keys()) == expected_class_names + ['accuracy', 'macro avg', 'weighted avg']
    for metric in ['accuracy', 'macro avg', 'weighted avg']:
        assert metric in results_json["classification_report"].keys()




def test_logistic_regression_binary(ds):
    my_lr = MyLogisticRegression(ds=ds, y_col="is_bot", param_grid=logistic_params_grid)
    assert my_lr.n_classes == 2
    my_lr.train_eval()
    results_ok(my_lr)

def test_logistic_regression_multiclass(ds):
    my_lr = MyLogisticRegression(ds=ds, y_col="fourway_label", param_grid=logistic_params_grid)
    assert my_lr.n_classes == 4
    my_lr.train_eval()
    results_ok(my_lr)



def test_random_forest_binary(ds):
    pipeline = MyRandomForest(ds=ds, y_col="is_bot", param_grid=forest_params_grid)
    assert pipeline.n_classes == 2
    pipeline.train_eval()
    results_ok(pipeline)

def test_random_forest_multiclass(ds):
    pipeline = MyRandomForest(ds=ds, y_col="fourway_label", param_grid=forest_params_grid)
    assert pipeline.n_classes == 4
    pipeline.train_eval()
    results_ok(pipeline)
