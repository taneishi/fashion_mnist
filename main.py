import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
import timeit
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
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28*28)

    return images, labels

def load_data(path='data/fashion'):
    ### Loading the dataset...

    X_main, y_main = load_mnist(path, kind='train')
    X_main = X_main.astype(np.float32)
    y_main = y_main.astype(np.float32)

    X_test, y_test = load_mnist(path, kind='t10k')
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    print('%s train shape %s test shape %s' % (path, X_main.shape, X_test.shape))

    return X_main, y_main, X_test, y_test

def preprocess(X, y):
    ### Standard scaling the pixel values with mean=0.0 and var=1.0

    sc = StandardScaler()
    X_std = sc.fit_transform(X)

    ### Splitting the train dataset into train and validation sets

    X_train, X_valid, y_train, y_valid = train_test_split(X_std, y_main, test_size=0.1)

    return X_train, X_valid, y_train, y_valid

def train(X_train, X_valid, y_train, y_valid, args):
    ### Training XgBoost classifier on the dataset

    param_list = vars(args).copy()
    del param_list['early_stopping'], param_list['n_rounds']
    d_train = xgb.DMatrix(X_train, label=y_train)
    d_val = xgb.DMatrix(X_valid, label=y_valid)
    eval_list = [(d_train, 'train'), (d_val, 'validation')]

    start_time = timeit.default_timer()
    bst = xgb.train(param_list, d_train, args.n_rounds, evals=eval_list, 
            early_stopping_rounds=args.early_stopping, verbose_eval=50)
    print('%6.3f sec' % (timeit.default_timer() - start_time))

    return bst

def predict(X_test, y_test, bst):
    ### Standard scaling the test-sets for both datasets

    sc = StandardScaler()
    X_test_std = sc.fit_transform(X_test)

    ### Predicting with trained classifiers

    d_test = xgb.DMatrix(data=X_test_std)
    y_pred = bst.predict(d_test)

    ### Checking accuracy for fashion and MNIST datasets respectively

    acc = np.sum(y_pred == y_test) / y_test.shape
    print('accuracy %5.2f%%' % (acc * 100.))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eta', default=0.08, type=float)
    parser.add_argument('--max_depth', default=6, type=int)
    parser.add_argument('--subsample', default=0.8, type=float)
    parser.add_argument('--colsample_bytree', default=0.8, type=float)
    parser.add_argument('--objective', default='multi:softmax', type=str)
    parser.add_argument('--eval_metric', default='merror', type=str)
    parser.add_argument('--alpha', default=8, type=int)
    parser.add_argument('--lambda', default=2, type=int)
    parser.add_argument('--num_class', default=10, type=int)
    parser.add_argument('--n_rounds', default=600, type=int)
    parser.add_argument('--early_stopping', default=50, type=int)
    args = parser.parse_args()

    print(vars(args))

    X_main, y_main, X_test, y_test = load_data('data/fashion')
    X_train, X_valid, y_train, y_valid = preprocess(X_main, y_main)

    bst = train(X_train, X_valid, y_train, y_valid, args)
    predict(X_test, y_test, bst)

    X_main, y_main, X_test, y_test = load_data('data/mnist')
    X_train, X_valid, y_train, y_valid = preprocess(X_main, y_main)

    bst = train(X_train, X_valid, y_train, y_valid, args)
    predict(X_test, y_test, bst)
