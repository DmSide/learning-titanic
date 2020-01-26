import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer


if __name__ == '__main__':
    salary_names = ['FullDescription','LocationNormalized','ContractTime','SalaryNormalized']
    data_train = pandas.read_csv(
        'salary-train.csv',
        skiprows=1,
        nrows=10000,
        names=salary_names)
    data_test = pandas.read_csv(
        'salary-test-mini.csv',
        skiprows=1,
        nrows=10000,
        names=salary_names)
    y = data_train['SalaryNormalized']
    del data_train['SalaryNormalized']
    del data_test['SalaryNormalized']
    print("Readed")
    data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    data_train['FullDescription'] = data_train['FullDescription'].to_string(na_rep='').lower()
    data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    data_test['FullDescription'] = data_test['FullDescription'].to_string(na_rep='').lower()

    print("FullDescription")
    vectorizer = TfidfVectorizer(min_df=5)
    X_train_vec = vectorizer.fit_transform(data_train['FullDescription'])
    X_test_vec = vectorizer.transform(data_test['FullDescription'])
    print("fit FullDescription")
    # Work with 'LocationNormalized', 'ContractTime'
    data_train['LocationNormalized'].fillna('nan', inplace=True)
    data_train['ContractTime'].fillna('nan', inplace=True)

    data_test['LocationNormalized'].fillna('nan', inplace=True)
    data_test['ContractTime'].fillna('nan', inplace=True)

    print("ContractTime")
    enc = DictVectorizer()
    X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
    X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
    print("fit ContractTime")

    a = 3

    # Объедините все полученные признаки в одну матрицу "объекты-признаки".
    # Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными.
    # Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
    # 3. Обучите гребневую регрессию с параметрами alpha=1 и random_state=241.
    # Целевая переменная записана в столбце SalaryNormalized.
    #
    # 4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv.
    # Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.

    # X_for_train = hstack([X_train_vec, X_train_categ])
    #
    # X_for_test = hstack([X_test_vec, X_test_categ])
    #
    # ridge = Ridge(alpha=, random_state=)
    #
    # ridge.fit(X_for_train, y_train)
    # ridge.predict(X_for_train, y_train)
    # Не забудьте округлить результаты и перевести их в str.