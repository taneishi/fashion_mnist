# fashion_mnist

Performance comparison of XgBoost when applied to MNIST and Fashion-MNIST datasets

```
{'eta': 0.08, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'objective': 'multi:softmax', 'eval_metric': 'merror', 'alpha': 8, 'lambda': 2, 'num_class': 10, 'n_rounds': 60, 'early_stopping': 50}
data/fashion train shape (60000, 784) test shape (10000, 784)
80.275 sec
accuracy 86.26%
data/mnist train shape (60000, 784) test shape (10000, 784)
76.100 sec
accuracy 94.52%
```
