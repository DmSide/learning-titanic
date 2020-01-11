import pandas

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

WINE_HEADERS = [
    'Class',
    'Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280 / OD315 of diluted wines',
    'Proline'
]

if __name__ == '__main__':
    df = pandas.read_csv('wine.data', names=WINE_HEADERS)
    print(df)
