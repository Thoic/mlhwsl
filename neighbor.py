import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    euclidean_distances,
    roc_auc_score,
)
import math
from scipy.spatial import distance

INPUT_FILE = "data/winequality-white.csv"


def euclidean_distance(p1, p2):
    return distance.euclidean(list(p1), list(p2))


def predict(x, features, target):
    distances = []
    for i in features:
        # print(i)
        distance = euclidean_distances([i, x])
        # print(distance)
        distances.append(distance)
    nearest = target[np.argmin(distances)]
    return nearest


df = pd.read_csv(INPUT_FILE, header=0, sep=";")

# pre-processing
# valb_idx = int(len(df.index)*0.6)
testb_idx = int(len(df.index) * 0.8)

X_train, y_train = df.iloc[:testb_idx, :-1], df.iloc[:testb_idx, -1]
# X_val, y_val = df.values[valb_idx:testb_idx, :-1], df.values[valb_idx:testb_idx, -1]
X_test, y_test = df.iloc[testb_idx:, :-1], df.iloc[testb_idx:, -1]

# modify train/test set for binary classification
y_train[:] = [1 if item >= 6 else -1 for item in y_train]
# y_val[:] = [1 if item >= 6 else -1 for item in y_val]
y_test[:] = [1 if item >= 6 else -1 for item in y_test]

X_train = list(X_train.to_records(index=False))
X_train = [list(i) for i in X_train]
y_train = y_train.tolist()
X_test = list(X_test.to_records(index=False))
X_test = [list(i) for i in X_test]
y_test = y_test.tolist()

predictions = []

for i in X_test:
    predictions.append(predict(i, X_train, y_train))

ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Nearest Neighbor")

naive_auc = roc_auc_score(y_test, predictions)
RocCurveDisplay.from_predictions(y_test, predictions)
plt.title("Nearest Neighbor ROC")

plt.show()
