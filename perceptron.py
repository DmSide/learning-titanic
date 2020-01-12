import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn_pandas import DataFrameMapper

if __name__ == '__main__':
    df_train = pandas.read_csv('perceptron-train.csv', header=None, names=['c1', 'c2', 'c3'])
    df_test = pandas.read_csv('perceptron-test.csv', header=None, names=['c1', 'c2', 'c3'])
    y_train = df_train['c1']
    X_train = df_train.copy()
    del X_train['c1']
    y_test = df_test['c1']
    X_test = df_test.copy()
    del X_test['c1']

    clf = Perceptron(random_state=241, max_iter=5, tol=None)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)

    ans1 = accuracy_score(y_test, y_pred_test)
    print(ans1)

    # mapper = DataFrameMapper([(X_train.columns, StandardScaler())])
    #
    # X_train_features = mapper.fit_transform(X_train.copy(), 4)
    # scaled_X_train = pandas.DataFrame(X_train_features, index=X_train.index, columns=X_train.columns)
    #
    # X_test_features = mapper.fit_transform(X_test.copy(), 4)
    # scaled_X_test = pandas.DataFrame(X_test_features, index=X_test.index, columns=X_test.columns)

    # y_train =
    # np.asarray(scaled_df_train['c1'], dtype="|S6")  # scaled_df_train.iloc[:, [0]].astype(float)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    #  y_test = np.asarray(scaled_df_test['c1'], dtype="|S6")  # scaled_df_test.iloc[:, [0]].astype(float)
    # IMPORTANT TO USE TRASFORM INSTAED OF FIT_TRANSFORM
    X_test = scaler.transform(X_test)

    # clf = Perceptron(random_state=241)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)

    ans2 = accuracy_score(y_test, y_pred_test)
    print(ans2)

    print(ans2-ans1)