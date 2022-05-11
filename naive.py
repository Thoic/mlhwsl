import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
import math

INPUT_FILE = "data/winequality-white.csv"

def bayes_predict(X_train, y_train, X_test):
    m, n = X_train.shape
    m_o, _ = X_test.shape

    m_pos = np.count_nonzero(y_train == 1.)
    m_neg = np.count_nonzero(y_train == -1.)

    probs_pos = np.full(m_o, np.log(m_pos / m))
    probs_neg = np.full(m_o, np.log(m_neg / m))

    sum_pos = np.zeros(m_o)
    sum_neg = np.zeros(m_o)

    x_pos = np.zeros((m_pos, n))
    x_neg = np.zeros((m_neg, n))
    idx_pos = 0
    idx_neg = 0
    for x, y in zip(X_train, y_train):
        if y == 1.:
            x_pos[idx_pos] = x
            idx_pos += 1
        else:
            x_neg[idx_neg] = x
            idx_neg += 1
    for i in range(n):
        # mean and variance
        mean_ipos = (1/m_pos)*np.sum(x_pos[:,i])
        var_ipos = (1/(m_pos-1))*np.sum((x_pos[:,i]-mean_ipos)**2)

        mean_ineg = (1/m_neg)*np.sum(x_neg[:,i])
        var_ineg = (1/(m_neg-1))*np.sum((x_neg[:,i]-mean_ineg)**2)

        # gauss probability
        a_i = X_test[:,i]
        probs_ipos = (1/(np.sqrt(2*np.pi*var_ipos)))*np.exp(-((a_i-mean_ipos)**2)/(2*var_ipos))
        probs_ineg = (1/(np.sqrt(2*np.pi*var_ineg)))*np.exp(-((a_i-mean_ineg)**2)/(2*var_ineg))
        sum_pos += np.log(probs_ipos)   
        sum_neg += np.log(probs_ineg)
    
    probs_tpos = probs_pos + sum_pos
    probs_tneg = probs_neg + sum_neg
    out = np.column_stack((probs_tneg, probs_tpos))
    y_predict = [1. if np.argmax(prob) == 1 else -1. for prob in out]


    return y_predict


def NaiveBayes():
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


    # x.fit(X_train, y_train)
    y_predict = bayes_predict(X_train, y_train, X_test)
    # print(preds)

    ConfusionMatrixDisplay.from_predictions(y_test, y_predict)
    plt.title("Naive Bayes")
    plt.savefig('bayes_confuse.png')

    naive_auc = roc_auc_score(y_test, y_predict)
    RocCurveDisplay.from_predictions(y_test, y_predict)
    plt.title("Naive Bayes Classifier ROC")
    x = np.arange(0,1.1, 0.1)
    plt.plot(x, x, '--', color='gray')

    plt.savefig('naive_bayes.png')

if __name__ == "__main__":
    NaiveBayes()