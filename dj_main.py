import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import numpy as np
from numpy import corrcoef

if __name__ == '__main__':
    dj_close_prices = pandas.read_csv('dj_close_prices.csv')
    # Remove date column
    dj_close_prices = dj_close_prices.iloc[:, 1:]

    jonse = pandas.read_csv('dj_djia_index.csv')
    # Remove date column
    jonse = jonse.iloc[:, 1:]

    X = np.array(dj_close_prices)
    # learn PCA
    pca = PCA(n_components=10)
    pca.fit(X)
    # Count dispersion > 90 %
    count = 0
    sum = 0
    for i in range(len(pca.explained_variance_ratio_)):
        count += 1
        value = pca.explained_variance_ratio_[i]
        sum += value
        if sum > 0.9:
            break
    print("Need components %d" % count)
    # Get only first conponent
    first_comp = pandas.DataFrame(pca.transform(X)[:, 0])

    coef = np.corrcoef(first_comp.T, jonse.T)[1, 0]
    print("Corrcoef %0.2f" % coef)

    indx = -1
    value = -1
    for i in range(len(pca.components_[0])):
        if value < pca.components_[0][i]:
            value = pca.components_[0][i]
            indx = i
    print("Company name: '%s' weight: %0.2f" % (dj_close_prices.columns[indx], value))
