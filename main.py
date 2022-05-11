import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

INPUT_FILE = "data/winequality-white.csv"

# perceptron learning rate
ETA = 0.1
MAX_ITER = 1000

# create a confusion matrix and determine tpr, fpr
def confuse_matrix(y_predict, y_test):
    conf_mat = [[0,0],
                [0,0]]
    for h, c in zip(y_predict, y_test):
        if h == -1. and c == -1.:
            conf_mat[0][0] += 1
        elif h == 1. and c == -1.:
            conf_mat[0][1] += 1
        elif h == -1. and c == 1.:
            conf_mat[1][0] += 1
        elif h == 1. and c == 1.:
            conf_mat[1][1] += 1
    
    tpr = conf_mat[1][1]/(conf_mat[1][1]+conf_mat[1][0])
    fpr = conf_mat[0][1]/(conf_mat[0][0]+conf_mat[0][1])

    return conf_mat, tpr, fpr

def output(w: np.ndarray, x: np.ndarray):
    o = np.dot(w, x)
    if (o) > 0:
        return 1.0
    else:
        return -1.0


def perceptronUpdate(w, t, o, x):
    for i in range(len(w)):
        w[i] += ETA * (t - o) * x[i]


def Perceptron(X_train, y_train, X_test, y_test):

    X_train = np.insert(X_train, 0, 1, axis=1)
    X_test = np.insert(X_test, 0, 1, axis=1)

    w = np.full(len(X_train[0]), 0.1)

    w_prev = w.copy()
    for j in range(0, MAX_ITER):
        for x, t in zip(X_train, y_train):
            o = output(w, x)
            perceptronUpdate(w, t, o, x)

        # if weights did not update
        if (w == w_prev).all():
            break
        else:
            w_prev = w.copy()

    y_predict = np.empty_like(y_test)
    for idx, item in enumerate(X_test):
        y_predict[idx] = output(w, item)

    mat, tpr, fpr = confuse_matrix(y_predict, y_test)
    print(mat)
    print(f'tpr: {tpr}, fpr:{fpr}')

    return y_predict


def LogReg(X_train, y_train, X_test, y_test):
    reg = LogisticRegression(max_iter=MAX_ITER)
    reg.fit(X_train, y_train)

    y_predict = reg.predict(X_test)

    mat, tpr, fpr = confuse_matrix(y_predict, y_test)
    print(pd.crosstab(y_test, y_predict, rownames=['Actual'], colnames=['Predicted']))
    print(f'tpr: {tpr}, fpr:{fpr}')

    return y_predict



def MLPClass(X_train, y_train, X_test, y_test):
    reg = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=0)
    reg.fit(X_train, y_train)

    y_predict = reg.predict(X_test)
    
    mat, tpr, fpr = confuse_matrix(y_predict, y_test)
    print(pd.crosstab(y_test, y_predict, rownames=['Actual'], colnames=['Predicted']))
    print(f'tpr: {tpr}, fpr:{fpr}')

    return y_predict



def NaiveBayes(X_train, y_train, X_test, y_test):
    reg = GaussianNB()
    reg.fit(X_train, y_train)

    y_predict = reg.predict(X_test)

    mat, tpr, fpr = confuse_matrix(y_predict, y_test)
    print(pd.crosstab(y_test, y_predict, rownames=['Actual'], colnames=['Predicted']))
    print(f'tpr: {tpr}, fpr:{fpr}')

    return y_predict

# classifier that predicts all entries to be 1
def naive_classifier(X_train, y_train, X_test, y_test):
    y_predict = np.full(len(y_test), 1)

    mat, tpr, fpr = confuse_matrix(y_predict, y_test)
    print(pd.crosstab(y_test, y_predict, rownames=['Actual'], colnames=['Predicted']))
    print(f'tpr: {tpr}, fpr:{fpr}')

    return y_predict

    

def main():
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

    print("Logistical Regression:")
    log_predict = LogReg(X_train, y_train, X_test, y_test)

    print('Perceptron: ')
    percept_predict = Perceptron(X_train, y_train, X_test, y_test)

    print("MLPClassifier: ")
    network_predict = MLPClass(X_train, y_train, X_test, y_test)

    print("Naive Bayes:")
    bayes_predict = NaiveBayes(X_train, y_train, X_test, y_test)

    print("Naive Classifier: ")
    naive_predict = naive_classifier(X_train, y_train, X_test, y_test)

    percept_auc = roc_auc_score(y_test, percept_predict)
    log_auc = roc_auc_score(y_test, log_predict)
    network_auc = roc_auc_score(y_test, network_predict)
    bayes_auc = roc_auc_score(y_test, bayes_predict)
    naive_auc = roc_auc_score(y_test, naive_predict)


    print(f'percept_auc:{percept_auc}, log_auc:{log_auc}, network_auc:{network_auc}, bayes_auc:{bayes_auc}, naive_auc:{naive_auc}')



if __name__ == "__main__":
    main()
