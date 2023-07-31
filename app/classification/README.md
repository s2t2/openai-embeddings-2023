
### Classification

Perform classification on the fully dimensional data. Binary classification on bot label, botometer labels, and opinion score. Multiclass classification on fourway labels (opinion x bot status).


#### Logistic Regression


```sh
python -m app.classification.logistic_regression

# K_FOLDS=10 python -m app.classification.logistic_regression
```

#### Random Forest

```sh
python -m app.classification.random_forest

# K_FOLDS=10 python -m app.classification.random_forest
```
