import pandas
import numpy as np
from sklearn.svm import SVC

if __name__ == '__main__':
    df = pandas.read_csv('svm-data.csv', header=None)
    y = df.loc[:, 0]
    X = df.loc[:, 1:2]
    sv = SVC(C=100000, random_state=241, kernel='linear')
    ans = sv.fit(X, y)

    print( sv.support_)
    print(sv.support_vectors_)