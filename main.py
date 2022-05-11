import numpy as np
import pandas as pd
from logistic import LogisticRegression
from percept import Perceptron
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from util import confuse_matrix

INPUT_FILE = "data/winequality-white.csv"

# perceptron learning rate
ETA = 0.1
MAX_ITER = 1000

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
    log_predict = LogisticRegression(X_train, y_train, X_test, y_test, MAX_ITER, ETA)

    print("Perceptron: ")
    percept_predict = Perceptron(X_train, y_train, X_test, y_test)

    print("Naive Bayes:")
    bayes_predict = NaiveBayes(X_train, y_train, X_test, y_test)

    # print("Naive Classifier: ")
    # naive_predict = naive_classifier(X_train, y_train, X_test, y_test)

    # percept_auc = roc_auc_score(y_test, percept_predict)
    # RocCurveDisplay.from_predictions(y_test, percept_predict)
    # plt.title("Perceptron ROC")

    # log_auc = roc_auc_score(y_test, log_predict)
    # RocCurveDisplay.from_predictions(y_test, log_predict)
    # plt.title("Logistic Regression ROC")

    # network_auc = roc_auc_score(y_test, network_predict)
    # RocCurveDisplay.from_predictions(y_test, network_predict)
    # plt.title("Multilayer Perceptron ROC")

    # bayes_auc = roc_auc_score(y_test, bayes_predict)
    # RocCurveDisplay.from_predictions(y_test, bayes_predict)
    # plt.title("Naive Bayes ROC")

    # naive_auc = roc_auc_score(y_test, naive_predict)
    # RocCurveDisplay.from_predictions(y_test, naive_predict)
    # plt.title("Naive Classifier ROC")

    # print(
    #     f"percept_auc:{percept_auc}, log_auc:{log_auc}, network_auc:{network_auc}, bayes_auc:{bayes_auc}, naive_auc:{naive_auc}"
    # )

    # plt.show()


if __name__ == "__main__":
    main()
