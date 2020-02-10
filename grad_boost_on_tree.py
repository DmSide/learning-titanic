from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from  sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
#
# n_estimators
# learning_rate
# verbose
# staged_decision_function()
# train_test_split
#
# X_train, X_test, y_train, y_test =
#                      train_test_split(X, y,
#                                       test_size=0.33,
#                                       random_state=42)
#
# pred = clf.predict_proba(X_test)
#
#
#
# plt.figure()
# plt.plot(test_loss, 'r', linewidth=2)
# plt.plot(train_loss, 'g', linewidth=2)
# plt.legend(['test', 'train'])


def sigmoid(x):
    # return 1.0 / 1 + math.exp(-x)
    return 1.0 / (1 + np.exp(x))


if __name__ == '__main__':
    data = pd.read_csv('gbm-data.csv')
    np_data = data.values
    y = np_data[:, 0]
    X = np_data[:, 1:]
    # X = data.drop('Activity', axis=1).values
    # y = data.Activity.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)
    for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
        gbc = GradientBoostingClassifier(
            learning_rate=learning_rate,
            n_estimators=250,
            verbose=True,
            random_state=241)
        gbc.fit(X=X_train, y=y_train)
        staged_decision_train = gbc.staged_decision_function(X_train)
        # staged_decision_test = gbc.staged_decision_function(X_test)
        # yy1 = sigmoid(np.array(list(staged_decision_train)))
        # yy2 = sigmoid(staged_decision_test)
        test_loss = list()
        for i, y_pred in enumerate(staged_decision_train):
            y_pred = 1.0 / (1.0 - np.exp(- y_pred))
            test_loss.append([i + 1, log_loss(y_test, y_pred)])
        test_loss = pd.DataFrame(test_loss, columns=['iter', 'loss'])

        # Извлечение значения минимального лосса
        test_loss[test_loss.loss == test_loss.loss.min()]
    print(1)