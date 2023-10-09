
### Classification

Perform classification on the fully dimensional data. Binary classification on bot label, botometer labels, and opinion score. Multiclass classification on fourway labels (opinion x bot status).


Each job will generate a CSV file of predictions. Which we are uploading to Google Drive under "text-embedding-ada-002/results/classifications/`TARGET_NAME`/`MODEL_NAME`/predictions.csv" for further analysis of any mis-classifications / confusions.

#### Logistic Regression


```sh
python -m app.classification.logistic_regression

FIG_SHOW=true FIG_SAVE=true python -m app.classification.logistic_regression
```

#### Decision Tree

```sh
python -m app.classification.decision_tree
```

#### Random Forest

```sh
python -m app.classification.random_forest
```


#### XGBoost


```sh
python -m app.classification.xgboost
```
