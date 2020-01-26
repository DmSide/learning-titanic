import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer


if __name__ == '__main__':
    data_test = pandas.read_csv('salary-test-mini.csv')
    data_train = pandas.read_csv('salary-train.csv')
    data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    data_train['FullDescription'] = data_train['FullDescription'].to_string(na_rep='').lower()
    data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    data_test['FullDescription'] = data_test['FullDescription'].to_string(na_rep='').lower()
    vectorizer = TfidfVectorizer(min_df=5)
    X_train_descr = vectorizer.fit_transform(data_train['FullDescription'])
    X_test_descr = vectorizer.transform(data_test['FullDescription'])
    # Work with 'LocationNormalized', 'ContractTime'
    data_train['LocationNormalized'].fillna('nan', inplace=True)
    data_train['ContractTime'].fillna('nan', inplace=True)
    data_test['LocationNormalized'].fillna('nan', inplace=True)
    data_test['ContractTime'].fillna('nan', inplace=True)
    enc = DictVectorizer()
    X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
    X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
    a=3

    # Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'. Код для этого был приведен выше.
    # Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
    # Объедините все полученные признаки в одну матрицу "объекты-признаки". Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными. Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
    # 3. Обучите гребневую регрессию с параметрами alpha=1 и random_state=241. Целевая переменная записана в столбце SalaryNormalized.
    #
    # 4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv. Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.