# AUC-ROC Area Under ROC-Curve
from sklearn.metrics import roc_auc_score


import numpy as np
import pandas
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns
import math


def sigmoid(x):
    # return 1.0 / 1 +  + math.exp(-x)
    return 1.0 / (1 + math.exp(-x))


def distance(a, b):
    return np.sqrt(np.square(a[0]-b[0]) + np.square((a[1]-b[1])))


def log_regression(X, y, k, w, C, epsilon, max_iter):
    w1, w2 = w
    for i in range(max_iter):
        w1new = w1 + k * np.mean(y * X[:, 0] * (1 - (1. / (1 + np.exp(-y * (w1 * X[:, 0] + w2 * X[:, 1])))))) - k * C * w1
        w2new = w2 + k * np.mean(
            y * X[:, 1] * (1 - (1. / (1 + np.exp(-y * (w2 * X[:, 1] + w2 * X[:, 1])))))) - k * C * w2

        if distance((w1new, w2new), (w1, w2)) < epsilon:
            break
        w1, w2 = w1new, w2new

    predictions = []

    for i in range(len(X)):
        t1 = -w1 * X[i, 0] - w2 * X[i, 1]
        s = sigmoid(t1)
        predictions.append(s)

    return predictions


if __name__ == '__main__':
    df = pandas.read_csv('data-logistic.csv', header=None)
    print(df)
    y = df.values[:, :1].T[0]
    X = df.values[:, 1:]

    # FOR PLOTS
    # sns.set_context("notebook", font_scale=1.1)
    # sns.set_style("ticks")
    # sns.regplot('X', 'Y', data=X, logistic=True)
    # plt.ylabel('Probability')
    # plt.xlabel('Explanatory')

    logistic = LogisticRegression()
    logistic.fit(X, y)
    logistic.score(X, y)
    print('Coefficient: \n', logistic.coef_)
    print('Intercept: \n', logistic.intercept_)
    print('R² Value: \n', logistic.score(X, y))

    p0 = log_regression(X, y, 0.1, [0.0, 0.0], 0, 0.00001, 10000)
    p1 = log_regression(X, y, 0.1, [0.0, 0.0], 10, 0.00001, 10000)

    print(roc_auc_score(y, p0))
    print(roc_auc_score(y, p1))

