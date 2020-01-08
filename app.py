import pandas

if __name__ == '__main__':
    data = pandas.read_csv('titanic.csv', index_col='PassengerId')
    print(data)
