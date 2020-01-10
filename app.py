import pandas


def get_first_name(full_name):
    try:
        return full_name.split(',')[0]
    except Exception as ex:
        return None


def get_middle_name(full_name):
    try:
        return full_name.split(',')[1].split('.')[0]
    except Exception as ex:
        return None


def get_last_name(full_name):
    try:
        mn = full_name.split(',')[1].split('.')[0].strip()
        if mn not in ['Mrs', 'Miss']:
            return None
        return full_name.split(',')[1].split('.')[1]
    except Exception as ex:
        return None


def get_last_name2(full_name):
    try:
        fn = full_name.split(',')[1].split('.')[1]
        if ' ' in fn:
            return fn.split()[0]
    except Exception as ex:
        return None


def get_last_name3(full_name):
    try:
        mn = full_name.split(',')[1].split('.')[0].strip()
        fn = full_name.split(',')[1].split('.')[1]
        if mn not in ['Mrs', 'Miss']:
            return None
        if ' ' in fn:
            return fn.split()[0]
    except Exception as ex:
        return None


if __name__ == '__main__':
    df = pandas.read_csv('titanic.csv', index_col='PassengerId')
    # print(df.head())
    # data_sex = pandas.read_csv('titanic.csv', usecols=["Sex"], squeeze=True)
    all_passenges_count = len(df)
    # Count number of males and females on the ship
    # df = df.set_index("Sex")
    # man_count = len(df.loc["male"])
    # man_count = data_sex[data_sex.apply(lambda x: x == "male")].count()
    # woman_count = len(df.loc["female"])
    # woman_count = data_sex[data_sex.apply(lambda x: x == "female")].count()
    print("all_passenges_count", all_passenges_count)
    print("man_count", len(df[df["Sex"] == "male"]))
    print("woman_count", len(df[df["Sex"] == "female"]))
    # Survived_count
    survived_count = len(df[df["Survived"] == 1])
    print("survived_count", survived_count)
    print("survived_proc", 100*survived_count/all_passenges_count)
    # First class
    first_class_count = len(df[df["Pclass"] == 1])
    print("first_class_count", first_class_count)
    print("first_class_proc", 100*first_class_count/all_passenges_count)
    # Age
    print("age_median",  df["Age"].median())
    print("age_mean", df["Age"].mean())
    # Corr Pirson
    print(f"pir_corr by Parch and SibSp {df.corr()['SibSp']['Parch']}")
    # Name
    data_name = df["Name"]
    # df[['Last_name', 'First_name']]=data_name.str.split(',', expand=True)
    # data_name.str.partition(',', True)
    last_name_frame = data_name.apply(get_last_name)
    last_name_frame = last_name_frame.dropna()
    aa = [x.replace("((().,\"", "").split() for x in last_name_frame]
    l = list()
    for a in aa:
        l.extend(a)
    llll = pandas.Series(l)
    # print(last_name_frame.value_counts()[:20])
    print(llll.value_counts()[:40])

