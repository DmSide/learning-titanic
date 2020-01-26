import pandas
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    df = pandas.read_csv('salary-test-mini.csv')
    train = pandas.read_csv('salary-train.csv')
    train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    train['FullDescription'] = train['FullDescription'].to_string(na_rep='').lower()
    vectorizer = TfidfVectorizer(min_df=5)
    X_vect = vectorizer.fit_transform(train['FullDescription'])
    a=3