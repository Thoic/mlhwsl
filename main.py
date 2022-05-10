import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

INPUT_FILE = 'data/winequality-white.csv'

# perceptron learning rate
ETA = 0.1
MAX_ITER = 1000

def output(w: np.ndarray, x: np.ndarray):
    o = np.dot(w, x)
    if (o) > 0:
        return 1.
    else:
        return -1.

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

        print(f'after the {j} iteration')
        print(f'w: {w}')

        # if weights did not update
        if (w == w_prev).all():
            break
        else:
            w_prev = w.copy()
    
    y_predict = np.empty_like(y_test)
    for idx, item in enumerate(X_test):
        y_predict[idx] = output(w, item)
    
    confuse_matrix = pd.crosstab(y_test, y_predict, rownames=['Actual'], colnames=['Predicted'])
    print(confuse_matrix)
    

def LogReg(X_train, y_train, X_test, y_test):
    reg = LogisticRegression(max_iter=MAX_ITER)
    reg.fit(X_train, y_train)

    predict = reg.predict(X_test)

    confuse_matrix = pd.crosstab(y_test, predict, rownames=['Actual'], colnames=['Predicted'])
    print(confuse_matrix)
    

def main():
    df = pd.read_csv(INPUT_FILE, header=0, sep=';')

    # pre-processing
    # valb_idx = int(len(df.index)*0.6)
    testb_idx = int(len(df.index)*0.8)

    X_train, y_train = df.values[:testb_idx, :-1], df.values[:testb_idx, -1]
    # X_val, y_val = df.values[valb_idx:testb_idx, :-1], df.values[valb_idx:testb_idx, -1]
    X_test, y_test = df.values[testb_idx:, :-1], df.values[testb_idx:, -1]

    # modify train/test set for binary classification
    y_train[:] = [1 if item >= 6 else -1 for item in y_train]
    # y_val[:] = [1 if item >= 6 else -1 for item in y_val]
    y_test[:] = [1 if item >= 6 else -1 for item in y_test]

    # LinReg(X_train, y_train, X_test, y_test)
    LogReg(X_train, y_train, X_test, y_test)
    # Perceptron(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()