import pandas

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
