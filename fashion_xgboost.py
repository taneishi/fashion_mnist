import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def load_mnist(path, kind='train'):
    import gzip
    import os

    '''Load MNIST data from `path`'''
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels

def load_data(path='data/fashion'):
    ### Loading the dataset...

    X_main, y_main = load_mnist(path, kind='train')
    X_main = X_main.astype(np.float32)
    y_main = y_main.astype(np.float32)

    X_test, y_test = load_mnist(path, kind='t10k')
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    print (X_main.shape, y_main.shape)
    print (X_test.shape, y_test.shape)

    return X_main, y_main, X_test, y_test

def preprocess(X, y, sc):
    ### Standard scaling the pixel values with mean=0.0 and var=1.0

    X_std = sc.fit_transform(X)

    ### Splitting the train dataset into train and validation sets

    X_train, X_valid, y_train, y_valid = train_test_split(X_std, y_main, test_size=0.1)

    return X_train, X_valid, y_train, y_valid

def train(X_train, X_valid, y_train, y_valid):
    ### Training XgBoost classifier on the dataset

    param_list = [('eta', 0.08), ('max_depth', 6), ('subsample', 0.8), ('colsample_bytree', 0.8),
            ('objective', 'multi:softmax'), ('eval_metric', 'merror'), ('alpha', 8), ('lambda', 2), ('num_class', 10)]
    n_rounds = 10 #600
    early_stopping = 50
        
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_val = xgb.DMatrix(X_valid, label=y_valid)
    eval_list = [(d_train, 'train'), (d_val, 'validation')]
    bst = xgb.train(param_list, d_train, n_rounds, evals=eval_list, early_stopping_rounds=early_stopping, verbose_eval=True)

    return bst

def predict(X_test, y_test, sc, bst):
    ### Standard scaling the test-sets for both datasets

    X_test_std = sc.fit_transform(X_test)

    ### Predicting with trained classifiers

    d_test = xgb.DMatrix(data=X_test_std)
    y_pred = bst.predict(d_test)

    ### Checking accuracy for fashion and MNIST datasets respectively

    return np.sum(y_pred == y_test) / y_test.shape

if __name__ == '__main__':
    X_main, y_main, X_test, y_test = load_data('data/fashion')
    X_mnist, y_mnist, X_test_mnist, y_test_mnist = load_data('data/mnist')

    sc = StandardScaler()
    X_train, X_valid, y_train, y_valid = preprocess(X_main, y_main, sc)
    X_mn_train, X_mn_valid, y_mn_train, y_mn_valid = preprocess(X_mnist, y_mnist, sc)

    bst = train(X_train, X_valid, y_train, y_valid)
    bst = train(X_mn_train, X_mn_valid, y_mn_train, y_mn_valid)

    acc = predict(X_test, y_test, sc, bst)
    print(acc)
    acc = predict(X_test_mnist, y_test_mnist, sc, bst)
    print(acc)
