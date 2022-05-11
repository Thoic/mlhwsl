import numpy as np
import pandas as pd
from util import confuse_matrix

def sigmoid(s):
    return 1/(1 + np.exp(-s))


def log_fit(X_train, y_train, MAX_ITER, ETA):
    m, n = X_train.shape
    w = np.full((n), 0)

    w_prev = w.copy()
    for iter in range(MAX_ITER):
        sum = 0
        for x, y in zip(X_train, y_train):
            numerator = np.dot(y, x)
            sum += numerator / (1 + np.exp(y*w.T*x))
        gradient = (-1/m)*sum

        w = w - ETA*gradient

        # if weights did not update
        diff = np.abs(w - w_prev)
        change = np.sum(diff)
        if (change < 10 ** -6):
            break
        else:
            w_prev = w.copy()

    return w

def log_predict(x, w):
    s = np.dot(w, x)
    if (sigmoid(s)) > 0:
        return 1.0
    else:
        return -1.0


def LogisticRegression(X_train, y_train, X_test, y_test, MAX_ITER, ETA):
    weights = log_fit(X_train, y_train, MAX_ITER, ETA)

    y_predict = np.empty_like(y_test)
    for idx, item in enumerate(X_test):
        y_predict[idx] = log_predict(weights, item)    

    mat, tpr, fpr = confuse_matrix(y_predict, y_test)
    print(pd.crosstab(y_test, y_predict, rownames=['Actual'], colnames=['Predicted']))
    print(f'tpr: {tpr}, fpr:{fpr}')

    return y_predict