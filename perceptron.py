import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
if __name__ == '__main__':
    df_train = pandas.read_csv('perceptron-train.csv', names=['c1', 'c2', 'c3'])
    df_test = pandas.read_csv('perceptron-test.csv', names=['c1', 'c2', 'c3'])
    y_train = df_train.iloc[:,[0]]
    X_train = df_train.iloc[:,1:]
    y_test = df_test.iloc[:, [0]]
    X_test = df_test.iloc[:, 1:]
    clf = Perceptron(random_state=241)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)

    ans1 = accuracy_score(y_test, y_pred_test)
    print(ans1)

    mapper = pandas.DataFrameMapper([(df_train.columns, StandardScaler())])

    df_train_features = mapper.fit_transform(df_train.copy(), 4)
    scaled_df_train = pandas.DataFrame(df_train_features, index=df_train.index, columns=df_train.columns)

    df_test_features = mapper.fit_transform(df_test.copy(), 4)
    scaled_df_test = pandas.DataFrame(df_test_features, index=df_test.index, columns=df_test.columns)

    y_train = scaled_df_train.iloc[:, [0]]
    X_train = scaled_df_train.iloc[:, 1:]
    y_test = scaled_df_test.iloc[:, [0]]
    X_test = scaled_df_test.iloc[:, 1:]

    clf = Perceptron(random_state=241)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)

    ans2 = accuracy_score(y_test, y_pred_test)
    print(ans2)

    print(ans2-ans1)