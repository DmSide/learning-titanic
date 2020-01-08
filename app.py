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
    print(df.head())
    # data_sex = pandas.read_csv('titanic.csv', usecols=["Sex"], squeeze=True)

    # Count number of males and females on the ship
    data_sex = df["Sex"]
    all_passenges_count = data_sex.count()
    man_count = data_sex[data_sex.apply(lambda x: x == "male")].count()
    woman_count = data_sex[data_sex.apply(lambda x: x == "female")].count()
    print("all_passenges_count", all_passenges_count)
    print("man_count", man_count)
    print("woman_count", woman_count)
    # Survived_count
    data_survived = df["Survived"]
    survived_count = data_survived[data_survived.apply(lambda x: x == 1)].count()
    print("survived_count", survived_count)
    print("survived_proc", 100*survived_count/all_passenges_count)
    # First class
    data_class = df["Pclass"]
    first_class_count = data_class[data_class.apply(lambda x: x == 1)].count()
    print("first_class_count", first_class_count)
    print("first_class_proc", 100*first_class_count/all_passenges_count)
    # Age
    data_age = df["Age"]
    age_median = data_age.median()
    age_mean = data_age.mean()
    print("age_median", age_median)
    print("age_mean", age_mean)
    # Corr Pirson
    pir1 = df.corr()['Parch']['SibSp']
    pir2 = df.corr()['SibSp']['Parch']
    print(pir1, '   ', pir2)
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

