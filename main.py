import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

input_file = 'data/winequality-white.csv'

# perceptron learning rate
ETA = 0.1

def output(w: np.ndarray, x: np.ndarray):
    if (np.dot(w, x.T)) > 0:
        return 1
    else:
        return -1

def perceptronUpdate(w, t, o, x):
    for i in range(len(w)):
        w[i] += ETA * (t - o) * x[i]

def Perceptron(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame):

    X_train.insert(0, 'convenience', 1)
    X_test.insert(0, 'convenience', 1)

    print(X_train)

    w = np.zeros(len(X_train.to_numpy()[0]))

    w_prev = w.copy()
    for j in range(1, 1001):
        for x, t in zip(X_train.to_numpy(), y_train.to_numpy()):
            o = output(w, x)
            perceptronUpdate(w, t, o, x)

        print(f'after the {j} iteration')
        print(f'w: {w}')

        # if weights did not update
        if (w == w_prev).all():
            break
        else:
            w_prev = w.copy()
    
    y_predict = []
    for item in X_test.iloc:
        y_predict.append(output(w, item))
    
    confuse_matrix = pd.crosstab(y_test, y_predict, rownames=['Actual'], colnames=['Predicted'])
    print(confuse_matrix)
    

def LinReg(X_train, y_train, X_test, y_test):
    # linear regression
    reg = LinearRegression().fit(X_train, y_train)

    predict = reg.predict(X_test)
    print(predict)

    confuse_matrix = pd.crosstab(y_test, predict, rownames=['Actual'], colnames=['Predicted'])

def LogReg(X_train, y_train, X_test, y_test):
    reg = LogisticRegression().fit(X_train, y_train)

    predict = reg.predict(X_test)

    confuse_matrix = pd.crosstab(y_test, predict, rownames=['Actual'], colnames=['Predicted'])
    print(confuse_matrix)
    

def main():
    df = pd.read_csv(input_file, header=0, sep=';')

    # pre-processing
    testb_idx = int(len(df.index)*0.8)

    X_train, y_train = df.iloc[:testb_idx, :-1], df.iloc[:testb_idx, -1]

    X_test, y_test = df.iloc[testb_idx:, :-1], df.iloc[testb_idx:, -1]

    # modify train/test set for binary classification
    y_train[:] = [1 if item >= 6 else -1 for item in y_train]
    y_test[:] = [1 if item >= 6 else -1 for item in y_test]

    # LinReg(X_train, y_train, X_test, y_test)
    # LogReg(X_train, y_train, X_test, y_test)
    Perceptron(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()