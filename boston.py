# Play with Minkovsky coefficient
# Boston data
#  https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
#
from sklearn import datasets, preprocessing, neighbors
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import matplotlib.pyplot as plt


VERSION_SCLEARN_MORE_THEN_0_18_1 = True

if __name__ == '__main__':
    data = datasets.load_boston()
    X = data['data']
    y = data['target']
    X = preprocessing.scale(X)
    scoring = ['mean_squared_error', 'neg_mean_squared_error'][VERSION_SCLEARN_MORE_THEN_0_18_1]

    show_plots = True

    best_result = -100000
    best_result_i = 0
    best_results = []

    for p in np.linspace(start=1, stop=10, num=200):
        model = neighbors.KNeighborsRegressor(
            metric='minkowski',
            p = p,
            n_neighbors=5,
            weights='distance'
        )
        model.fit(X, y)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        results = cross_val_score(model, X, y, cv=kf, scoring=scoring)
        results_avg = np.average(results)
        best_results.append(results_avg)
        if results_avg > best_result:
            best_result = results_avg
            best_result_i = p

    if show_plots:
        plt.plot(best_results)
        plt.show()

    print(best_result, best_result_i, np.array(best_results).max())