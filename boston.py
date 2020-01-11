# Play with Minkovsky coefficient
# Boston data
#  https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
#
from sklearn import datasets, preprocessing, neighbors

VERSION_SCLEARN_MORE_THEN_0_18_1 = True

if __name__ == '__main__':
    data = datasets.load_boston()
    X = data['data']
    Y = data['target']
    X = preprocessing.scale(X)
    scoring = ['mean_squared_error', 'neg_mean_squared_error'][VERSION_SCLEARN_MORE_THEN_0_18_1]

    model = neighbors.KNeighborsRegressor(
        metric='minkowski',
        p=1,
        n_neighbors=5,
        weights='distance'
    )
    # numpy.linspace ???
