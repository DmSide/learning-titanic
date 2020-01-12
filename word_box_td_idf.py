import numpy as np
# import pandas
# import heapq
# from scipy.sparse import csr_matrix
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
    cv = KFold(n_splits=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241, C=best_c)
    ans = clf.fit(X_vect, y)
    # DONT WORK
    # print(clf.coef_)
    # coeffs = csr_matrix.toarray(clf.coef_)
    # coeffs_list = coeffs.tolist()[0]
    # inds = np.argpartition(coeffs, -10)[-10:]
    # top10 = heapq.nlargest(10, range(len(coeffs_list)), coeffs_list.__getitem__)

    coefs = abs(clf.coef_.todense().A1)
    coefs = np.argsort(coefs)
    top10 = coefs[-10:]

    top10.sort()
    words = [vectorizer.get_feature_names()[x] for x in top10]
    print(words)