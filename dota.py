import json
import bz2
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np


def sigmoid(x):
    # return 1.0 / 1 + math.exp(-x)
    return 1.0 / (1 + np.exp(x))


def distance(a, b):
    return np.sqrt(np.square(a[0]-b[0]) + np.square((a[1]-b[1])))


def log_regression(X, y, k, w, C, epsilon, max_iter):
    w1, w2 = w
    for i in range(max_iter):
        w1new = w1 + k * np.mean(
            y * X[:, 0] * (1 - (1. / (1 + np.exp(-y * (w1 * X[:, 0] + w2 * X[:, 1])))))) - k * C * w1
        w2new = w2 + k * np.mean(
            y * X[:, 1] * (1 - (1. / (1 + np.exp(-y * (w1 * X[:, 0] + w2 * X[:, 1])))))) - k * C * w2

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
    dota_features = pd.read_csv("dota_features.csv")
    X_test = pd.read_csv("dota_features_test.csv")
    dota_features.fillna(value=0, inplace=True)
    X_test.fillna(value=0, inplace=True)
    # Chapter 1. Grad Boost
    # 1.1 Get full list of values with NaN/Null. What does it mean?
    # 1.2 Name of target variable row
    # radiant_win
    # 1.3 How long and what quality was on boost with 30 trees?
    # 30.42s
    # 1.2892
    # 1.4 Should we use more than 30 trees? How can we speed up the method?

    # Remove future values
    del dota_features['duration']
    del dota_features['tower_status_radiant']
    del dota_features['tower_status_dire']
    del dota_features['barracks_status_dire']
    del dota_features['barracks_status_radiant']

    y_train = dota_features['radiant_win']
    X_train = dota_features
    del X_train['radiant_win']

    gbc = GradientBoostingClassifier(
        # learning_rate=learning_rate,
        n_estimators=30,  # 250,
        verbose=True,
        random_state=241)
    gbc.fit(X=X_train, y=y_train)
    ans = gbc.predict(X=X_test)

    # 2. Logistic regression
    # 2.1 Value of log regression. What is faster: log regression or grad boost?
    # 2.2 What if we delete some rows?
    # 2.3 How many different categories of hero in data?
    # 2.4 What if we add "word bag"? Is it better?
    # 2.5 Min and Max predicted value

    logistic = LogisticRegression()
    logistic.fit(X_train, y_train)
    logistic.score(X_train, y_train)
    print('Coefficient: \n', logistic.coef_)
    print('Intercept: \n', logistic.intercept_)
    print('R² Value: \n', logistic.score(X, y))

    p0 = log_regression(X_train, y_train, 0.1, [0.0, 0.0], 0, 0.00001, 10000)
    p1 = log_regression(X_train, y_train, 0.1, [0.0, 0.0], 10, 0.00001, 10000)

    print(f'{roc_auc_score(y, p0):.3f}')
    print(f'{roc_auc_score(y, p1):.3f}')  # 0.937
    
    print('Done')
