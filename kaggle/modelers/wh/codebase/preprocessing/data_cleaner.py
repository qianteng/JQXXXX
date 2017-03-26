from __future__ import print_function
import utils.conf as conf
import pandas as pd
from sklearn import preprocessing


def read_data():
    train = pd.read_csv(conf.train_data_path, sep=',')
    test = pd.read_csv(conf.test_data_path, sep=',')
    return train, test


def impute(df):
    df['Upc'] = df.fillna(0)
    df['FinelineNumber'] = df['FinelineNumber'].fillna(-1).astype('int')


# Encode categorical variables
def encode(df, columns, col=None, dict=None):
    # Auto encoding
    for col_name in columns:
        le = preprocessing.LabelEncoder()
        le.fit(df[col_name])
        df['Encoded_' + col_name] = le.transform(df[col_name])

    # encode by dictionary
    if (col is not None) & (dict is not None):
        for key, value in dict.items():
            df[col].replace(key, value, inplace=True)
        df['Encoded_' + col] = df[col]


def clean():
    train, test = read_data()
    impute(train)
    impute(test)
    wd_dict = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    encode(train, ['DepartmentDescription', 'Upc'], col='Weekday', dict=wd_dict)
    encode(test, ['DepartmentDescription', 'Upc'], col='Weekday', dict=wd_dict)
    train.drop(['Weekday', 'Upc', 'DepartmentDescription'], axis=1, inplace=True)
    test.drop(['Weekday', 'Upc', 'DepartmentDescription'], axis=1, inplace=True)
    return train, test
