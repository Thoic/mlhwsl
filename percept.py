import numpy as np
from util import confuse_matrix

def output(w: np.ndarray, x: np.ndarray):
    o = np.dot(w, x)
    if (o) > 0:
        return 1.0
    else:
        return -1.0


def perceptronUpdate(w, t, o, x, ETA):
    for i in range(len(w)):
        w[i] += ETA * (t - o) * x[i]


def Perceptron(X_train, y_train, X_test, y_test, ETA, MAX_ITER):

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