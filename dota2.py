
import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("features.csv", index_col="match_id")
print(train.head())

#РЈРґР°Р»РёС‚Рµ РїСЂРёР·РЅР°РєРё, СЃРІСЏР·Р°РЅРЅС‹Рµ СЃ РёС‚РѕРіР°РјРё РјР°С‚С‡Р° (РѕРЅРё РїРѕРјРµС‡РµРЅС‹ РІ РѕРїРёСЃР°РЅРёРё РґР°РЅРЅС‹С… РєР°Рє РѕС‚СЃСѓС‚СЃС‚РІСѓСЋС‰РёРµ РІ С‚РµСЃС‚РѕРІРѕР№ РІС‹Р±РѕСЂРєРµ).
train.drop([
    "duration",
    "tower_status_radiant",
    "tower_status_dire",
    "barracks_status_radiant",
    "barracks_status_dire",
], axis=1, inplace=True)

count_na = len(train) - train.count()
count_na[count_na > 0].sort_values(ascending=False) / len(train)

train.fillna(0, inplace=True)

X_train = train.drop("radiant_win", axis=1)
y_train = train["radiant_win"]

cv = KFold(n_splits=5, shuffle=True, random_state=42)


def score_gb(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}

    for n_estimators in [10, 20, 30, 50, 100, 250]:
        print(f"n_estimators={n_estimators}")
        model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)

        start_time = datetime.datetime.now()
        score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        print(f"Score: {score:.3f}")
        print(f"Time elapsed: {datetime.datetime.now() - start_time}")

        scores[n_estimators] = score
        print()

    return pd.Series(scores)

scores = score_gb(X_train, y_train)
scores.plot()


#2 Logistic
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)


def score_lr(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}

    for i in range(-5, 6):
        C = 10.0 ** i

        print(f"C={C}")
        model = LogisticRegression(C=C, random_state=42)

        start_time = datetime.datetime.now()
        score = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        print(f"Score: {score:.3f}")
        print(f"Time elapsed: {datetime.datetime.now() - start_time}")

        scores[i] = score
        print()

    return pd.Series(scores)

scores = score_lr(X_train, y_train)
scores.plot()

def print_best_lr_score(scores: pd.Series):
    best_iteration = scores.sort_values(ascending=False).head(1)
    best_C = 10.0 ** best_iteration.index[0]
    best_score = best_iteration.values[0]

    print(f"РќР°РёР»СѓС‡С€РµРµ Р·РЅР°С‡РµРЅРёРµ РїРѕРєР°Р·Р°С‚РµР»СЏ AUC-ROC РґРѕСЃС‚РёРіР°РµС‚СЃСЏ РїСЂРё C = {best_C:.2f} Рё СЂР°РІРЅРѕ {best_score:.2f}.")

    print_best_lr_score(scores)

hero_columns = [f"r{i}_hero" for i in range (1, 6)] + [f"d{i}_hero" for i in range (1, 6)]
cat_columns = ["lobby_type"] + hero_columns
X_train.drop(cat_columns, axis=1, inplace=True)

scores = score_lr(X_train, y_train)
scores.plot()

print_best_lr_score(scores)

unique_heroes = np.unique(train[hero_columns].values.ravel())
N = max(unique_heroes)
print(f"Р§РёСЃР»Рѕ СѓРЅРёРєР°Р»СЊРЅС‹С… РіРµСЂРѕРµРІ РІ train: {len(unique_heroes)}. РњР°РєСЃРёРјР°Р»СЊРЅС‹Р№ ID РіРµСЂРѕСЏ: {N}.")

def get_pick(data: pd.DataFrame) -> pd.DataFrame:
    X_pick = np.zeros((data.shape[0], N))

    for i, match_id in enumerate(data.index):
        for p in range(1, 6):
            X_pick[i, data.loc[match_id, f"r{p}_hero"] - 1] = 1
            X_pick[i, data.loc[match_id, f"d{p}_hero"] - 1] = -1

    return pd.DataFrame(X_pick, index=data.index, columns=[f"hero_{i}" for i in range(N)])

X_pick = get_pick(train)
X_pick.head()

X_train = pd.concat([X_train, X_pick], axis=1)

scores = score_lr(X_train, y_train)
scores.plot()

print_best_lr_score(scores)


model = LogisticRegression(C=0.1, random_state=42)
model.fit(X_train, y_train)

test = pd.read_csv("data/features_test.csv", index_col="match_id")
test.fillna(0, inplace=True)

X_test = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
X_test.drop(cat_columns, axis=1, inplace=True)
X_test = pd.concat([X_test, get_pick(test)], axis=1)
X_test.head()

preds = pd.Series(model.predict_proba(X_test)[:, 1])
preds.describe()

preds.plot.hist(bins=30)