# fashion_mnist

Performance comparison of XgBoost when applied to MNIST and Fashion-MNIST datasets

```
{'eta': 0.08, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'alpha': 8, 'lambda': 2, 'num_class': 10, 'n_rounds': 600, 'early_stopping': 50}
data/fashion train shape (60000, 784) test shape (10000, 784)
[0]     train-merror:0.17772    validation-merror:0.19817
Multiple eval metrics have been passed: 'validation-merror' will be used for early stopping.

Will train until validation-merror hasn't improved in 50 rounds.
[50]    train-merror:0.10228    validation-merror:0.13133
[100]   train-merror:0.07313    validation-merror:0.11533
[150]   train-merror:0.05181    validation-merror:0.10567
[200]   train-merror:0.03528    validation-merror:0.10250
[250]   train-merror:0.02461    validation-merror:0.09950
[300]   train-merror:0.01709    validation-merror:0.09667
[350]   train-merror:0.01189    validation-merror:0.09617
[400]   train-merror:0.00865    validation-merror:0.09483
Stopping. Best iteration:
[374]   train-merror:0.01028    validation-merror:0.09417

512.162 sec
accuracy 89.75%
data/mnist train shape (60000, 784) test shape (10000, 784)
[0]     train-merror:0.14659    validation-merror:0.16250
Multiple eval metrics have been passed: 'validation-merror' will be used for early stopping.

Will train until validation-merror hasn't improved in 50 rounds.
[50]    train-merror:0.03904    validation-merror:0.04883
[100]   train-merror:0.02106    validation-merror:0.03567
[150]   train-merror:0.01204    validation-merror:0.02983
[200]   train-merror:0.00737    validation-merror:0.02750
[250]   train-merror:0.00533    validation-merror:0.02533
[300]   train-merror:0.00391    validation-merror:0.02433
[350]   train-merror:0.00306    validation-merror:0.02383
[400]   train-merror:0.00257    validation-merror:0.02350
Stopping. Best iteration:
[365]   train-merror:0.00293    validation-merror:0.02333

456.866 sec
accuracy 96.68%
```
