from __future__ import print_function
import utils.conf as conf
import pandas as pd
from sklearn import preprocessing


def read_data():
    train = pd.read_csv(conf.train_data_path, sep=',')
    test = pd.read_csv(conf.test_data_path, sep=',')
    return train, test


def impute(df):
    df['Upc'] = df['Upc'].fillna(0)
    df['FinelineNumber'] = df['FinelineNumber'].fillna(-1).astype('int')


# Encode categorical variables
def encode(df1, columns, df2=None, col=None, encode_dict=None):
    # Auto encoding
    for col_name in columns:
        le = preprocessing.LabelEncoder()
        le.fit(df1[col_name])
        df1['Encoded_' + col_name] = le.transform(df1[col_name])
        if df2 is not None:
            le.fit(df2[col_name])
            df2['Encoded_' + col_name] = le.transform(df2[col_name])

    # encode by dictionary
    if (col is not None) & (dict is not None):
        for key, value in encode_dict.items():
            df1[col].replace(key, value, inplace=True)
        df1['Encoded_' + col] = df1[col]
        if df2 is not None:
            for key, value in encode_dict.items():
                df2[col].replace(key, value, inplace=True)
            df2['Encoded_' + col] = df2[col]

def clean():
    train, test = read_data()
    impute(train)
    impute(test)
    wd_dict = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
    encode(train, ['DepartmentDescription', 'Upc'], df2=test, col='Weekday', encode_dict=wd_dict)
    #encode(test, ['DepartmentDescription', 'Upc'], col='Weekday', dict=wd_dict)
    train.drop(['Weekday', 'Upc', 'DepartmentDescription'], axis=1, inplace=True)
    test.drop(['Weekday', 'Upc', 'DepartmentDescription'], axis=1, inplace=True)
    return train, test
