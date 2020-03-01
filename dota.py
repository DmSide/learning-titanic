import json
import bz2
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

if __name__ == '__main__':
    dota_features = pd.read_csv("dota_features.csv", index_col="match_id")
    X_test = pd.read_csv("dota_features_test.csv")

    # Chapter 1. Grad Boost
    # 1.1 Get full list of values with NaN/Null. What does it mean?
    # dire_bottle_time, dire_courier_time
    # It means there are no event with this name occurs
    # 1.2 Name of target variable row
    # radiant_win
    # 1.3 How long and what quality was on boost with 30 trees?
    # 30.42s
    # 1.2892
    # 1.4 Should we use more than 30 trees? How can we speed up the method?

    # Remove future values
    dota_features.drop([
        "duration",
        "tower_status_radiant",
        "tower_status_dire",
        "barracks_status_radiant",
        "barracks_status_dire",
    ], axis=1, inplace=True)
    count_na = len(dota_features) - dota_features.count()
    count_na[count_na > 0].sort_values(ascending=False) / len(dota_features)
    dota_features.fillna(value=0, inplace=True)
    X_test.fillna(value=0, inplace=True)

    y_train = dota_features['radiant_win']
    X_train = dota_features.drop('radiant_win', axis=1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grad_boost_scores = {}
    for n_estimators in [10, 20, 30, 40, 50, 100, 150, 200, 250]:  # range(1, 250):
        gbc = GradientBoostingClassifier(
            # learning_rate=learning_rate,
            n_estimators=n_estimators,
            verbose=True,
            random_state=241)
        start_time = datetime.datetime.now()
        score = cross_val_score(gbc, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        print(f"Score: {score:.3f}")
        print(f"Time elapsed: {datetime.datetime.now() - start_time}")
        grad_boost_scores[n_estimators] = score
        # gbc.fit(X=X_train, y=y_train)
        # ans = gbc.predict(X=X_test)

    pd.Series(grad_boost_scores).plot()
    # 2. Logistic regression
    # 2.1 Value of log regression. What is faster: log regression or grad boost?
    # 2.2 What if we delete some rows?
    # 2.3 How many different categories of hero in data?
    # 2.4 What if we add "word bag"? Is it better?
    # 2.5 Min and Max predicted value
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)

    grad_boost_scores = {}

    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        print(f"C={C}")
        model = LogisticRegression(C=C, random_state=42)

        start_time = datetime.datetime.now()
        score = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        print(f"Score: {score:.3f}")
        print(f"Time elapsed: {datetime.datetime.now() - start_time}")

        grad_boost_scores[C] = score
        print()

    grad_boost_scores = pd.Series(grad_boost_scores)
    grad_boost_scores.plot()

    best_log_score = grad_boost_scores.sort_values(ascending=False).head(1)
    best_C = best_log_score.index[0]
    best_score = best_log_score.values[0]

    r_hero_columns = [f"r{i}_hero" for i in range(1, 6)]
    f_hero_columns = [f"d{i}_hero" for i in range(1, 6)]
    hero_columns = r_hero_columns + f_hero_columns
    unique_heroes = np.unique(X_train[hero_columns].values.ravel())
    N = max(unique_heroes)

    print(f"Unique_heroes {N}")
    print('Done')
