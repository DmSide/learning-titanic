import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, cross_validate

if __name__ == '__main__':
    # read dataset
    data = pd.read_csv('abalone.csv')
    # Tranform 'Sex' column to number value
    data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
    y = data['Rings']
    X = data.iloc[:, 0: -1]
    for i in range(1, 50 + 1):
        clf = RandomForestRegressor(random_state=1, n_estimators=i)
        model = clf.fit(X, y)
        estimator = KFold(n_splits=5, random_state=1, shuffle=True)
        # scoring = r2_score(y_true=, y_pred=)
        # cross_validate(model, X, y, cv=estimator)
        results = cross_val_score(model, X, y, cv=estimator, scoring='r2')
        # results_avg = np.average(results)
        results_avg = round(results.mean(), 2)
        # print(results_avg)
        if results_avg > 0.52:
            print(i, results_avg)

            with open('/home/dima/lr_w5_z1_1_1.txt', 'w') as out:
                out.write(str(i))
                # print(i, file=out, flush=True)
            break
       #  print(results)
