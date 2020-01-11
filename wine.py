# Analizing of data
# https://archive.ics.uci.edu/ml/datasets/Wine
# usuing k-neigbors Classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
import pandas

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import tree
import matplotlib.pyplot as plt

WINE_HEADERS = [
    'Class',
    'Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280 / OD315 of diluted wines',
    'Proline'
]

if __name__ == '__main__':
    df = pandas.read_csv('wine.data', names=WINE_HEADERS)
    y = df['Class']
    del df['Class']

    best_result = 0
    best_result_i = 0
    best_results = []

    for k in range(1, 51):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(df, y)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        results = cross_val_score(model, df, y, cv=kf, scoring="accuracy")
        results_avg = np.average(results)
        best_results.append(results_avg)
        if results_avg > best_result:
            best_result = results_avg
            best_result_i = k

    best_result_np = np.array(best_results).max()
    print(best_result, best_result_i, best_result_np)
    plt.plot(best_results)
    plt.show()
