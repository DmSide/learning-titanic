import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from numpy import corrcoef

if __name__ == '__main__':
    dj_close_prices = pandas.read_csv('dj_close_prices.csv')
    # Remove date column
    dj_close_prices = dj_close_prices.iloc[:, 1:]

    jonse = pandas.read_csv('dj_djia_index.csv')
    # Remove date column
    jonse = jonse.iloc[:, 1:]

    ipca = PCA(n_components=10)
    ipca.fit_transform(dj_close_prices)
    ipca.transform(dj_close_prices)