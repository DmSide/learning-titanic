from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
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
    # for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
    for learning_rate in [0.2]:
        gbc = GradientBoostingClassifier(
            learning_rate=learning_rate,
            n_estimators=250,
            verbose=True,
            random_state=241)
        gbc.fit(X=X_train, y=y_train)
        staged_decision_train = gbc.staged_decision_function(X_test)
        # staged_decision_test = gbc.staged_decision_function(X_test)
        # yy1 = sigmoid(np.array(list(staged_decision_train)))
        # yy2 = sigmoid(staged_decision_test)
        test_loss = np.empty(250)
        for i, y_pred in enumerate(staged_decision_train):
            y_pred = 1.0 / (1.0 + np.exp(- y_pred))
            test_loss[i] = log_loss(y_test, y_pred)
        print(test_loss.max())
        if learning_rate == 0.2:
            print('learning_rate == 0.2')
            lr02_min = test_loss.min()
            lr02_idxmin = test_loss.argmin()
            print(lr02_min)
            print(lr02_idxmin)
            with open('/home/dima/lr_w5_z2_1_1.txt', 'w') as out:
                out.write(f'overfitting')
            with open('/home/dima/lr_w5_z2_2_1.txt', 'w') as out:
                out.write(f'{lr02_min:.2f} {lr02_idxmin}')

            rfc = RandomForestClassifier(random_state=241, n_estimators=lr02_idxmin)
            rfc.fit(X=X_train, y=y_train)
            rfc_pred = rfc.predict_proba(X_test)
            ls_ans = log_loss(y_test, rfc_pred)
            with open('/home/dima/lr_w5_z2_3_1.txt', 'w') as out:
                out.write(f'{ls_ans:.2f}')
        else:
            plt.figure()
            plt.plot(test_loss, 'r', linewidth=2)
            plt.show()


    print(1)