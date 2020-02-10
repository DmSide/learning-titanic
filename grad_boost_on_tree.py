from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from  sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import pandas as pd
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

if __name__ == '__main__':
    pd.read_csv('gbm-data.csv')