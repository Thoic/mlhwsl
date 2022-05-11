import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from util import confuse_matrix
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score

INPUT_FILE = "data/winequality-white.csv"

# log learning rate
ETA = 0.1
MAX_ITER = 1000

THRESHOLD = 0.5

def sigmoid(s):
    return 1/(1 + np.exp(-s))


def log_fit(X_train, y_train):
    m, n = X_train.shape
    w = np.full((n), 0)

    w_prev = w.copy()
    for iter in range(MAX_ITER):
        sum = 0
        for x, y in zip(X_train, y_train):
            numerator = np.dot(y, x)
            sum += numerator / (1 + np.exp(y * w.T @ x))
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
    if (sigmoid(s)) > THRESHOLD:
        return 1.0
    else:
        return -1.0


def LogisticRegression():
    df = pd.read_csv(INPUT_FILE, header=0, sep=";")

    # pre-processing
    # valb_idx = int(len(df.index)*0.6)
    testb_idx = int(len(df.index) * 0.8)

    X_train, y_train = df.values[:testb_idx, :-1], df.values[:testb_idx, -1]
    # X_val, y_val = df.values[valb_idx:testb_idx, :-1], df.values[valb_idx:testb_idx, -1]
    X_test, y_test = df.values[testb_idx:, :-1], df.values[testb_idx:, -1]

    # modify train/test set for binary classification
    y_train[:] = [1 if item >= 6 else -1 for item in y_train]
    # y_val[:] = [1 if item >= 6 else -1 for item in y_val]
    y_test[:] = [1 if item >= 6 else -1 for item in y_test]

    weights = log_fit(X_train, y_train)

    y_predict = np.empty_like(y_test)
    for idx, item in enumerate(X_test):
        y_predict[idx] = log_predict(weights, item)    
    
    mat, tpr, fpr = confuse_matrix(y_predict, y_test)
    print(pd.crosstab(y_test, y_predict, rownames=['Actual'], colnames=['Predicted']))
    print(f'tpr: {tpr}, fpr:{fpr}')

    ConfusionMatrixDisplay.from_predictions(y_test, y_predict)
    plt.title("Logistic Regression")

    log_auc = roc_auc_score(y_test, y_predict)
    print(log_auc)
    RocCurveDisplay.from_predictions(y_test, y_predict)
    plt.title("Logistic Regression Classifier ROC")

    plt.savefig('log.png')

if __name__ == "__main__":
    LogisticRegression()