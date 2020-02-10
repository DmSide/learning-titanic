import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

if __name__ == '__main__':
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([-3, 1, 10])
    clf = RandomForestRegressor(n_estimators=100)
    clf.fit(X, y)
    predictions = clf.predict(X)
    print(predictions)



    r2_score([10, 11, 12], [9, 11, 12.1])