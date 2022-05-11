import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.naive_bayes import GaussianNB
import math

INPUT_FILE = "data/winequality-white.csv"


class NaiveBayes:
    def calc_statistics(self, features, target):
        self.mean = features.groupby(target).apply(np.mean).to_numpy()
        self.var = features.groupby(target).apply(np.var).to_numpy()

        return self.mean, self.var

    def gaussian_density(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp((-1 / 2) * ((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        prob = numerator / denominator
        return prob

    # prior probabilities
    def calc_prior(self, features, target):
        self.prior = (
            features.groupby(target).apply(lambda x: len(x)) / self.rows
        ).to_numpy()
        return self.prior

    # posterior probabilities
    def calc_posterior(self, x):
        posteriors = []
        for i in range(self.count):
            prior = np.log(self.prior[i])
            conditional = np.sum(np.log(self.gaussian_density(i, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def fit(self, features, target):
        # define class variables
        self.classes = np.unique(target)
        self.count = len(self.classes)
        self.feature_nums = features.shape[1]
        self.rows = features.shape[0]

        # calculate statistics
        self.calc_statistics(features, target)
        self.calc_prior(features, target)

    def predict(self, features):
        preds = [self.calc_posterior(f) for f in features.to_numpy()]
        return preds


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

x = NaiveBayes()
# print(type(X_train))
x.fit(X_train, y_train)
predict = x.predict(X_test)
# print(preds)

ConfusionMatrixDisplay.from_predictions(y_test, predict)
plt.title("Naive Bayes")

naive_auc = roc_auc_score(y_test, predict)
RocCurveDisplay.from_predictions(y_test, predict)
plt.title("Naive Bayes Classifier ROC")

plt.show()
