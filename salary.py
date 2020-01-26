import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge


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
    y_train = data_train['SalaryNormalized']
    del data_train['SalaryNormalized']
    del data_test['SalaryNormalized']
    print("Readed")
    temp_data_train_full_description = data_train['FullDescription']
    del data_train['FullDescription']
    temp_data_train_full_description = temp_data_train_full_description.replace('[^a-zA-Z0-9]', ' ', regex=True)
    temp_data_train_full_description = temp_data_train_full_description.str.lower()
    temp_data_test_full_description = data_test['FullDescription']
    del data_test['FullDescription']
    temp_data_test_full_description = temp_data_test_full_description.replace('[^a-zA-Z0-9]', ' ', regex=True)
    temp_data_test_full_description = temp_data_test_full_description.str.lower()

    print("FullDescription")
    vectorizer = TfidfVectorizer(min_df=5)
    X_train_vec = vectorizer.fit_transform(temp_data_train_full_description)
    X_test_vec = vectorizer.transform(temp_data_test_full_description)
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

    X_for_train = hstack([X_train_vec, X_train_categ])

    X_for_test = hstack([X_test_vec, X_test_categ])

    ridge = Ridge(alpha=1, fit_intercept=False, solver='lsqr')

    ridge.fit(X_for_train, y_train)
    rp = ridge.predict(X_for_test)
    print(rp)
    print(f'{rp[0]:.2f}')
    print(f'{rp[1]:.2f}')
    # Не забудьте округлить результаты и перевести их в str.