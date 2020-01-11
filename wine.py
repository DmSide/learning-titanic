# Analizing of data
# https://archive.ics.uci.edu/ml/datasets/Wine
# usuing k-neigbors Classifier
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
import pandas

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,  KFold
from sklearn.preprocessing import scale
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
    scaled_df = scale(df)
    show_plots = False

    for X in [df, scaled_df]:
        best_result = 0
        best_result_i = 0
        best_results = []

        for k in range(1, 51):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X, y)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            results = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
            results_avg = np.average(results)
            best_results.append(results_avg)
            if results_avg > best_result:
                best_result = results_avg
                best_result_i = k

        print(best_result, best_result_i)
        if show_plots:
            plt.plot(best_results)
            plt.show()
            # Close plot to continue

