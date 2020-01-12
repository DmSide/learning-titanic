import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
# Преобразование обучающей выборки нужно делать с помощью функции fit_transform, тестовой — с помощью transform
from sklearn.svm import SVC

if __name__ == '__main__':

    newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
    X = newsgroups.data
    y = newsgroups.target
    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)
    # y = vectorizer.transform(y) # DONT NEED TO TRANSFORM - ERROR
    # Find BEST C PARAMS
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(n_splits=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(X_vect, y)
    c13 = gs.best_params_
    print(c13)

    best_c = c13['C']
    # Continue withy best C patams
    sv = SVC(C=100000, random_state=241, kernel='linear')
    # ans = sv.fit(X, y)
    # print(sv.coef_)
    # feature_mapping = vectorizer.get_feature_names()
    # print(feature_mapping[i])