import json
import bz2
import datetime
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score


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
    dota_features = pd.read_csv("dota_features.csv", index_col="match_id")
    X_test = pd.read_csv("dota_features_test.csv")

    # Chapter 1. Grad Boost
    # 1.1 Get full list of values with NaN/Null. What does it mean?
    # dire_bottle_time, dire_courier_time
    # It means there are no event with this name occurs
    # 1.2 Name of target variable row
    # radiant_win
    # 1.3 How long and what quality was on boost with 30 trees?
    # 30.42s
    # 1.2892
    # 1.4 Should we use more than 30 trees? How can we speed up the method?

    # Remove future values
    dota_features.drop([
        "duration",
        "tower_status_radiant",
        "tower_status_dire",
        "barracks_status_radiant",
        "barracks_status_dire",
    ], axis=1, inplace=True)
    count_na = len(dota_features) - dota_features.count()
    count_na[count_na > 0].sort_values(ascending=False) / len(dota_features)
    dota_features.fillna(value=0, inplace=True)
    X_test.fillna(value=0, inplace=True)

    y_train = dota_features['radiant_win']
    X_train = dota_features.drop('radiant_win', axis=1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = {}
    for n_estimators in [10, 20, 30, 40, 50, 100, 150, 200, 250]:  # range(1, 250):
        gbc = GradientBoostingClassifier(
            # learning_rate=learning_rate,
            n_estimators=n_estimators,
            verbose=True,
            random_state=241)
        start_time = datetime.datetime.now()
        score = cross_val_score(gbc, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        print(f"Score: {score:.3f}")
        print(f"Time elapsed: {datetime.datetime.now() - start_time}")
        scores[n_estimators] = score
        # gbc.fit(X=X_train, y=y_train)
        # ans = gbc.predict(X=X_test)

    pd.Series(scores).plot()
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
