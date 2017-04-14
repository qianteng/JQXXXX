import os
import time
import pickle
import pandas as pd

curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, '../Data')

def get_file_path(file_name):
    curr_dir = os.getcwd()
    data_dir = os.path.join(curr_dir, '../Data')
    file_path = os.path.join(data_dir, file_name)
    return file_path

def pickle_dump(obj, out_path):
    fileobj = open(out_path, 'wb')
    pickle.dump(obj, fileobj)
    fileobj.close()
    return


def convert_weekday(train_df):
    dayOfWeek = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
    train_df['Weekday'] = train_df['Weekday'].map(dayOfWeek)
    return train_df


def apply_func_groupby_VisitNumber(df):
    TripType = df['TripType'].mean()
    Weekday = df['Weekday'].mean()
    Upc = df['Upc'].drop_duplicates().count()
    ScanCount = df['ScanCount'].sum()
    DepartmentDescription = df['DepartmentDescription'].drop_duplicates().count()
    FinelineNumber = df['FinelineNumber'].drop_duplicates().count()

    res = [TripType, Weekday, Upc, ScanCount, DepartmentDescription, FinelineNumber]
    res = pd.DataFrame(res).T
    res.columns = ['TripType', 'Weekday', 'Upc', 'ScanCount', 'DepartmentDescription', 'FinelineNumber']
    return res

def occurence_frequency_encoding(df):
    """
    to calculate the occurence frequency of values in categorical column in each TripType
    :param df: df[['TripType', 'col_name']]  
                col_name: Upc, DepartmentDescription, or FinelineNumber
                col_name must be categorical column
    :return: 
    """
    col_names = df.columns.tolist()
    if col_names[0] != 'TripType' and col_names[1] == 'TripType':
        col_names = col_names[::-1]

    encoding_map_df = df.groupby(col_names[::-1])[col_names[1]].agg({'Frequency': 'count'})
    encoding_map_df.reset_index(inplace=True)
    encoding_map_df = encoding_map_df.pivot(index=col_names[1], columns=col_names[0])['Frequency']
    encoding_map_df =  encoding_map_df.divide(encoding_map_df.sum(axis=1), axis=0)
    encoding_map_df.columns = [col_names[1]+'_'+str(col) for col in encoding_map_df.columns.tolist()]

    encoding_map_df.to_csv(os.path.join(data_dir, r"encoding_map_{_col_name}.csv".format(_col_name=col_names[1])))

    return encoding_map_df

def apply_func_groupby_VisitNumber(df):
    
    Weekday = df['Weekday'].mean()
    Upc = df['Upc'].drop_duplicates().count()
    ScanCount = df['ScanCount'].sum()
    DepartmentDescription = df['DepartmentDescription'].drop_duplicates().count()
    FinelineNumber = df['FinelineNumber'].drop_duplicates().count()
    
    res = [Weekday, Upc, ScanCount, DepartmentDescription, FinelineNumber]
    res = pd.DataFrame(res).T
    res.columns = ['Weekday', 'UpcCount', 'ScanCount', 'DepartmentCount', 'FinelineCount']

    return res

def get_detailed_ScanCounts_by_VisitNumber(train_df):
    pos_count = train_df[['VisitNumber', 'ScanCount']]
    neg_count = train_df[['VisitNumber', 'ScanCount']]

    pos_count.set_value(pos_count['ScanCount']<0, ['ScanCount'], 0)
    neg_count.set_value(neg_count['ScanCount']>0, ['ScanCount'], 0)

    pos_count = pos_count[['VisitNumber','ScanCount']].groupby('VisitNumber').sum()
    neg_count = neg_count[['VisitNumber','ScanCount']].groupby('VisitNumber').sum().apply(abs)

    abs_sum_count = train_df[['VisitNumber', 'ScanCount']].apply(abs).groupby('VisitNumber').sum()
    sum_count = train_df[['VisitNumber', 'ScanCount']].groupby('VisitNumber').sum()

    pos_count.columns = ['PosScanCount']
    neg_count.columns = ['NegScanCount']
    abs_sum_count.columns = ['AbsSumScanCount']
    sum_count.columns = ['SumScanCount']

    ScanCount_df = pos_count.join(neg_count, how='left').join(abs_sum_count, how='left').join(sum_count, how='left')

    pos_count.head()
    neg_count.head()
    abs_sum_count.head()
    sum_count.head()

    ScanCount_df.to_csv(os.path.join(curr_dir, 'ScanCount_df.csv'))

    return ScanCount_df

def encode_ScanCount_by_department_by_visitnumber(train_df):
    """
    :return: scancounts by department by visitnumber
    """
    train_df.set_index('VisitNumber', inplace=True)

    DepartmentDesc_map = train_df[['DepartmentDescription']].drop_duplicates().reset_index(drop=True).reset_index()
    DepartmentDesc_map.columns = ['departInd', 'DepartmentDescription']
    DepartmentDesc_map.set_index('DepartmentDescription', inplace=True)
    DepartmentDesc_map = DepartmentDesc_map.T.to_dict()
    train_df[['DepartmentDescription']] = train_df[['DepartmentDescription']].applymap(
        lambda x: DepartmentDesc_map[x]['departInd'])
    train_df.rename(columns={'DepartmentDescription': 'departInd'}, inplace=True)

    big_depart_counts = train_df[['departInd']].reset_index()
    big_depart_counts['CountsByDepart'] = train_df['ScanCount'].values

    big_depart_counts = big_depart_counts.groupby(['VisitNumber', 'departInd']).sum()
    big_depart_counts.reset_index(inplace=True)
    big_depart_counts = big_depart_counts.pivot(index='VisitNumber', columns='departInd').fillna(0)

    big_depart_counts.columns = ['depart_%d' % col for col in big_depart_counts.columns.levels[1]]

    return big_depart_counts

def process_ScanCount_cols(train_df):
    ScanCount_df = get_detailed_ScanCounts_by_VisitNumber()
    big_depart_counts = encode_ScanCount_by_department_by_visitnumber(train_df)
    train_df = train_df.join(ScanCount_df, how='left').join(big_depart_counts, how='left')
    return train_df

def get_big_df_train():
    train_csv_path = get_file_path('big_df_train.csv')
    big_df_train = pd.read_csv(train_csv_path)

    TripType_map = big_df_train['TripType'].drop_duplicates().sort_values().reset_index(drop=True).reset_index()
    TripType_map = TripType_map.rename(columns={'index': 'newTripType'})

    big_df_train = big_df_train.merge(TripType_map, left_on='TripType', right_on='TripType', how='left',
                                      suffixes=('', '_')).drop('TripType', axis=1)
    return big_df_train

def get_big_df_test():
    train_csv_path = get_file_path('big_df_test.csv')
    big_df_test = pd.read_csv(train_csv_path)
    return big_df_test


def big_df_pca_transform(XX_train, XX_test=None):
    from sklearn.decomposition import PCA
    nn = 0
    target_percent = 0.99
    explained_pct = 0
    while explained_pct < target_percent:
        nn += 1
        pca = PCA(n_components=nn)
        pca.fit(XX_train)
        explained_pct = sum(pca.explained_variance_ratio_)
    XX_train_pca = pca.transform(XX_train)

    if XX_test is not None:
        XX_test_pca = pca.transform(XX_test)
        return XX_train_pca, XX_test_pca
    else:
        return XX_train_pca


def run_feature_engineering(data_type):

    print('run_feature_engineering starting...')

    init_time = time.clock()

    df = pd.read_csv(get_file_path('{}.csv'.format(data_type)))
    df = convert_weekday(df)
    categorical_cols = ['Upc', 'DepartmentDescription', 'FinelineNumber']

    # get categorical column occurrence frequency encoding map
    if (1):
        print('occurence_frequency_encoding starting...')
        curr_time = time.clock()

        for col in categorical_cols:
            df = df[['TripType', col]]
            occurence_frequency_encoding(df=df)

        print('occurence_frequency_encoding finishing...', time.clock()-curr_time)

    # replace categorical columns with occur_freq encoded cols
    if (1):
        print('{}_df categorical cols replacing starting...'.format(data_type))
        curr_time = time.clock()

        for col in categorical_cols:
            csv_path = get_file_path(r"encoding_map_{_col_name}.csv".format(_col_name=col))
            encoding_map = pd.read_csv(csv_path)
            df = df.merge(encoding_map, left_on=col, right_on=col, how='left')

        df.drop(df[categorical_cols], axis=1, inplace=True)
        df_encoded_csv_path = os.path.join(data_dir, '{}_encoded.csv'.format(data_type))
        df.to_csv(df_encoded_csv_path, index=False)

        print('{}_df categorical cols replacing finish...'.format(data_type), time.clock()-curr_time)

    if (1):
        print('{}_df get detailed ScanCounts starting...'.format(data_type))
        curr_time = time.clock()
        df = process_ScanCount_cols(df)
        print('{}_df get detailed ScanCounts finish...'.format(data_type), time.clock()-curr_time)

    # get aggregate stats groupby VisitNumber
    if (1):
        print('get stats_groupby_VisitNumber_unencoded_{}.csv start...'.format(data_type))
        curr_time = time.clock()

        res = df.groupby(['VisitNumber']).apply(apply_func_groupby_VisitNumber)
        res.index = res.index.levels[0]
        res.to_csv(get_file_path(r'stats_groupby_VisitNumber_unencoded_{}.csv'.format(data_type)))
        print('get stats_groupby_VisitNumber_unencoded_{}.csv finish...'.format(data_type), time.clock()-curr_time)


    if (1):
        print('get big_df_{}.csv start...'.format(data_type))
        curr_time = time.clock()

        stats_groupby_VisitNumber_df = pd.read_csv(
            get_file_path(r'stats_groupby_VisitNumber_unencoded_{}.csv'.format(data_type)))
        big_df = df.merge(stats_groupby_VisitNumber_df, left_on='VisitNumber', right_on='VisitNumber', how='left',
                                suffixes=('', "_stats"))
        valid_cols = [col for col in big_df.columns if '_stats' not in col]
        big_df = big_df[valid_cols]
        big_df.to_csv(get_file_path(r'big_df_{}.csv'.format(data_type)), index=False)

        print('get big_df_{}.csv finish...'.format(data_type), time.clock()-curr_time)


    print('run_feature_engineering {_data_type} finish, total time: {_total_time}'.format(_data_type=data_type, _total_time=str(time.clock()-init_time)))

    return


if __name__ == '__main__':
    
    data_types = ['train', 'test']
    for dtype in data_types:
        curr_time = time.clock()
        print('{} start...'.format(dtype))
        run_feature_engineering(dtype)
        print('{} finish...'.format(dtype), time.clock()-curr_time)
    
    print('finish...')
        