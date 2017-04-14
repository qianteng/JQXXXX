import pandas as pd
from scipy import stats
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def aggregate_features_v1(df):
    # group by visit number
    groupby_vn = df.groupby('VisitNumber', as_index=True)
    # sum of scancount per visit
    df = df.join(groupby_vn['ScanCount'].sum(), on='VisitNumber', rsuffix='_sum_groupby_vn')
    # number of distinct department covered per visit
    df = df.join(groupby_vn.agg({'Encoded_DepartmentDescription': pd.Series.nunique}), on='VisitNumber',
                 rsuffix='_nunique_groupby_vn')
    # number of distinct fineline number covered per visit
    df = df.join(groupby_vn.agg({'FinelineNumber': pd.Series.nunique}), on='VisitNumber', rsuffix='_nunique_groupby_vn')
    # number of distinct UPC covered per visit
    df = df.join(groupby_vn.agg({'Upc' : pd.Series.nunique}), on='VisitNumber', rsuffix='_nunique_groupby_vn')
    # number of items purchased under a single fineline number per visit
    df = df.join(groupby_vn[['FinelineNumber']].agg(lambda x: stats.mode(x['FinelineNumber']).count[0]),
                 on='VisitNumber', rsuffix='_max_by_vn')
    # the fineline number under which the most items are purchased per visit
    df = df.join(groupby_vn[['FinelineNumber']].agg(lambda x: stats.mode(x['FinelineNumber'])[0]), on='VisitNumber',
                 rsuffix='_has_max_by_vn')
    # number of items purchased under a single department per visit
    df = df.join(groupby_vn[['Encoded_DepartmentDescription']]
                 .agg(lambda x: stats.mode(x['Encoded_DepartmentDescription']).count[0]), on='VisitNumber',
                 rsuffix='_max_by_vn')
    # the department under which the most items are purchased per visit
    df = df.join(groupby_vn[['Encoded_DepartmentDescription']]
                 .agg(lambda x: stats.mode(x['Encoded_DepartmentDescription'])[0]), on='VisitNumber',
                 rsuffix='_has_max_by_vn')

    # number of returned items per visit
    df = df.join(groupby_vn[['ScanCount']].agg(lambda x: -x[x < 0].sum()), on='VisitNumber',
                 rsuffix='_returned_items_by_vn')

    # bought items per visit
    df = df.join(groupby_vn[['ScanCount']].agg(lambda x: x[x > 0].sum()), on='VisitNumber',
                 rsuffix='_purchased_items_by_vn')
    return df


def aggregate_features_v2(df, quantile=0.3):

    # pick a list of fineline numbers that have high occurrences.
    fn_stat = df.groupby('FinelineNumber', as_index=False)['ScanCount'].sum()
    fn_list = fn_stat[fn_stat['ScanCount'] > fn_stat['ScanCount'].quantile(q=quantile)]['FinelineNumber']
    out = df[['VisitNumber']].drop_duplicates()
    # create dummies for finelien number
    count = 0
    for fn in fn_list:
        if count % 100 == 0:
            print("%d of %d fineline number features have been added." %(count, len(fn_list)))
        count += 1
        data = df[df['FinelineNumber'] == fn]
        data = data.groupby(['VisitNumber'], as_index=False)['ScanCount'].sum()
        data.rename(columns={'ScanCount': 'fn_%s' %(fn)}, inplace=True)
        out = out.merge(data, how='left', on=['VisitNumber'], copy=True)
        out['fn_%s' %(fn)].fillna(value=0, inplace=True)

    out = out.set_index('VisitNumber')
    # create dummies for department descriptions
    dummies = pd.get_dummies(df.DepartmentDescription)
    df[dummies.columns] = dummies
    df[dummies.columns] = df[dummies.columns].apply(lambda x: x * df['ScanCount'])
    groupby_vn = df.groupby('VisitNumber', as_index=True)

    #out = pd.DataFrame(index=df['VisitNumber'].unique())
    if 'TripType' in df.columns:
        out['TripType'] = groupby_vn['TripType'].max()
    out['ScanCountSum'] = groupby_vn['ScanCount'].sum()
    out['UPC_nunique'] = groupby_vn.agg({'Upc': pd.Series.nunique})
    out['FinelineNumber_max_num'] = groupby_vn[['FinelineNumber']].agg(lambda x: stats.mode(x['FinelineNumber']).count[0])
    out['FinelineNumber_has_max'] = groupby_vn[['FinelineNumber']].agg(lambda x: stats.mode(x['FinelineNumber'])[0])
    #out['Encoded_DepartmentDescription_max_num'] = groupby_vn[['Encoded_DepartmentDescription']].agg(
        #lambda x: stats.mode(x['Encoded_DepartmentDescription']).count[0])
    #out['Encoded_DepartmentDescription_has_max'] = groupby_vn[['Encoded_DepartmentDescription']].agg(
        #lambda x: stats.mode(x['Encoded_DepartmentDescription'])[0])
    out['Returned_num'] = groupby_vn[['ScanCount']].agg(lambda x: -x[x < 0].sum())
    out['Purchased_num'] = groupby_vn[['ScanCount']].agg(lambda x: x[x > 0].sum())

    for col_name in dummies.columns:
        out[col_name] = groupby_vn[[col_name]].agg(np.sum)

    return out


def generate_general_features(df):

    out = df[['VisitNumber']].drop_duplicates()

    # sum of scancount per visit
    data = df.groupby(['VisitNumber'], as_index=False)['ScanCount'].sum()
    data.rename(columns={'ScanCount': 'ScanCount_sum'}, inplace=True)
    out = out.merge(data, how='left', on='VisitNumber')

    # number of returned items per visit
    df['neg_count'] = df['ScanCount']
    df.ix[df['ScanCount'] > 0, 'neg_count'] = 0
    data = df.groupby(['VisitNumber'], as_index=False)['neg_count'].sum().abs()
    data.rename(columns={'neg_count': 'returned_items'}, inplace=True)
    out = out.merge(data, how='left', on='VisitNumber')

    # number of bought items per visit
    df['pos_count'] = df['ScanCount']
    df.ix[df['ScanCount'] < 0, 'pos_count'] = 0
    data = df.groupby(['VisitNumber'], as_index=False)['pos_count'].sum()
    data.rename(columns={'pos_count': 'bought_items'}, inplace=True)
    out = out.merge(data, how='left', on='VisitNumber')

    # min / max / mean / total number of items bought from each department  per visit
    data = df.groupby(['VisitNumber', 'DepartmentDescription'], as_index=False)['pos_count'].sum()
    data1 = data.groupby(['VisitNumber'], as_index=False)['pos_count'].min()
    data2 = data.groupby(['VisitNumber'], as_index=False)['pos_count'].max()
    data3 = data.groupby(['VisitNumber'], as_index=False)['pos_count'].mean()
    data1.rename(columns={'pos_count': 'Min'}, inplace=True)
    data2.rename(columns={'pos_count': 'Max'}, inplace=True)
    data3.rename(columns={'pos_count': 'Mean'}, inplace=True)
    out = out.merge(data1, how='left', on=['VisitNumber'], copy=True)
    out = out.merge(data2, how='left', on=['VisitNumber'], copy=True)
    out = out.merge(data3, how='left', on=['VisitNumber'], copy=True)
    out['Range'] = out['Max'] - out['Min']


    # number of distinct department covered per visit
    data = df.groupby(['VisitNumber', 'DepartmentDescription'], as_index=False)['ScanCount'].count()
    data = data.groupby(['VisitNumber'], as_index=False)['ScanCount'].count()
    data.rename(columns={'ScanCount': 'nunique_department'}, inplace=True)
    out = out.merge(data, how='left', on='VisitNumber', copy=True)

    # number of distinct fineline number covered per visit
    data = df.groupby(['VisitNumber', 'FinelineNumber'], as_index=False)['ScanCount'].count()
    data = data.groupby(['VisitNumber'], as_index=False)['ScanCount'].count()
    data.rename(columns={'ScanCount': 'nunique_fineline'}, inplace=True)
    out = out.merge(data, how='left', on='VisitNumber', copy=True)

    # number of distinct UPC covered per visit
    data = df.groupby(['VisitNumber', 'Upc'], as_index=False)['ScanCount'].count()
    data = data.groupby(['VisitNumber'], as_index=False)['ScanCount'].count()
    data.rename(columns={'ScanCount': 'nunique_UPC'}, inplace=True)
    out = out.merge(data, how='left', on='VisitNumber', copy=True)

    # max number of items purchased under a single fineline number per visit
    # the fineline number under which the most items are purchased per visit
    data = df.groupby(['VisitNumber', 'FinelineNumber'], as_index=False)['ScanCount'].sum()
    idx = data.groupby(['VisitNumber'], as_index=False)['ScanCount'].idxmax()
    data = data.ix[idx, :]
    data.rename(columns={'ScanCount': 'max_num_in_one_fineline', 'FinelineNumber': 'fl_has_max'}, inplace=True)
    out = out.merge(data, how='left', on='VisitNumber', copy=True)

    # the department under which the most items are purchased per visit
    # max number of items purchased under a single department per visit
    data = df.groupby(['VisitNumber', 'DepartmentDescription'], as_index=False)['ScanCount'].sum()
    idx = data.groupby(['VisitNumber'], as_index=False)['ScanCount'].idxmax()
    data = data.ix[idx, :]
    data.rename(columns={'ScanCount': 'max_num_in_one_department', 'DepartmentDescription':'dep_has_max'}, inplace=True)
    out = out.merge(data, how='left', on='VisitNumber', copy=True)

   # out['ratio_f_d'] = out['nunique_fineline'] / out['nunique_department']
   # out['ratio_u_d'] = out['nunique_UPC'] / out['nunique_department']
   # out['mean_to_min'] = out['Mean'] / out['Min']
   # out['mean_to_min'].replace('inf', 0, inplace=True)
    #out['max_to_mean'] = out['Max'] / out['Mean']
   # out['max_to_mean'].replace('inf', 0, inplace=True)
    out.drop('dep_has_max', axis=1, inplace=True)
    out = out.fillna(0)
    out.drop('VisitNumber', axis=1, inplace=True)
    feature_names = out.columns.values.tolist()

    return out, feature_names


def generate_sparse_categorical_features_bow(df, row_name, col_name, val_name, aggregate=False):
    out = df[[row_name]].drop_duplicates().set_index(row_name)
    dummies = pd.get_dummies(df[col_name])
    df[dummies.columns] = dummies
    if aggregate:
        df[dummies.columns] = df[dummies.columns].apply(lambda x: x * df[val_name])
        grouped = df.groupby(row_name, as_index=True)
        for col_name in dummies.columns:
            out[col_name] = grouped[[col_name]].agg(np.sum)
    else:
        grouped = df.groupby(row_name, as_index=True)
        for col in dummies.columns:
            out[col_name + str(col)] = grouped[[col]].first()
    return out, out.columns.values.tolist()


# Convert categoricla variable into dummy variables and return a dense matrix
def generate_dense_categorical_features_bow(df, row_name, col_name, val_name, aggregate=False, quantile=0.3):

    row_labels = df[row_name].unique()
    label2row = pd.Series(range(row_labels.size), index=row_labels)

    if aggregate:
        df = df.groupby([row_name, col_name], as_index=False)[val_name].sum()
    else:
        df = df.groupby([row_name, col_name], as_index=False)[val_name].count()

    col_stat = df.groupby(col_name, as_index=False)[val_name].sum()
    col_labels = col_stat[col_stat[val_name] > col_stat[val_name].quantile(q=quantile)][col_name]

    # col_labels = df[col_name].unique()
    label2col = pd.Series(range(col_labels.size), index=col_labels)

    feature_names = [col_name + '_' + str(i) for i in col_labels]

    cols = df[col_name].map(label2col)
    valid_mask = ~cols.isnull()
    rows = df[row_name].map(label2row)[valid_mask]
    vals = df[val_name][valid_mask]
    cols = cols[valid_mask]

    dense_matrix = sp.sparse.csc_matrix((vals, (rows, cols)), shape=(label2row.size, label2col.size))

    return dense_matrix, feature_names


# TF-IDF features generator
def generate_dense_categorical_features_tfidf(df,  col_name, count_name=None, group_by=None, sep='#', min_df=0.0, max_df=1.0,
                                              ngram_ragne=(1,1)):

    if count_name is not None:
        df['expanded_' + col_name] = df.apply(lambda x: '#'.join([x[col_name]]*abs(x[count_name])), axis=1)

    col2text = df[col_name]

    if group_by is not None:
        if count_name is not None:
            col_name = 'expanded_' + col_name
        col2text = df.groupby(group_by, as_index=True).agg({col_name: lambda x : sep.join(str(x))})

    def tokenizer(text): return text.split(sep)
    tfv = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_ragne, tokenizer=tokenizer)
    out = tfv.fit_transform(col2text[col_name])
    return out


def aggregate_features(df):
    general_features, general_feature_names = generate_general_features(df)

    fineline_features,  fl_feature_names = generate_dense_categorical_features_bow(df, 'VisitNumber', 'FinelineNumber',
                                                                               'ScanCount', aggregate=True, quantile=0.1
                                                                               )
    department_features, dp_feature_names = generate_dense_categorical_features_bow(df, 'VisitNumber',
                                                                                'DepartmentDescription', 'ScanCount',
                                                                                aggregate=True, quantile=0.0)
    upc_features, upc_feature_names = generate_dense_categorical_features_bow(df, 'VisitNumber', 'Upc', 'ScanCount',
                                                                          aggregate=True, quantile=0.1)
    weekday_features, wd_feature_names = generate_sparse_categorical_features_bow(df, 'VisitNumber', 'Weekday',
                                                                              'ScanCount')

    general_features = sp.sparse.csc_matrix(general_features.values)
    weekday_features = sp.sparse.csc_matrix(weekday_features.values)

    dept_tfidf = generate_dense_categorical_features_tfidf(df, 'DepartmentDescription', count_name='ScanCount',
                                                           group_by='VisitNumber').tocsc()

    fineline_tfidf = generate_dense_categorical_features_tfidf(df, 'FinelineNumber', group_by='VisitNumber',
                                                               min_df=0.3).tocsc()

    X = sp.sparse.hstack((general_features, fineline_features, weekday_features, department_features, dept_tfidf,
                          fineline_tfidf, upc_features))

    feature_names = general_feature_names + fl_feature_names + dp_feature_names + upc_feature_names + wd_feature_names

    # X = sp.sparse.hstack((fineline_features, department_features)).tocsc()
    # X = sp.sparse.hstack((general_features, sp.sparse.coo_matrix(np.ones((X.shape[0], 1))))).tocsc()

    return X, feature_names


def train_cross_validation(df):
    y = df.groupby(['VisitNumber'])['TripType'].first().values
    X, feature_names = aggregate_features(df)
    idx_train, idx_test = train_test_split(range(df['VisitNumber'].unique().shape[0]), test_size=0.33,
                                           random_state=32)
    y_train = y[idx_train]
    y_test = y[idx_test]
    X_train = X[idx_train]
    X_test = X[idx_test]
    return X_train, X_test, y_train, y_test, feature_names


def prepare_data(train, test):
    y = train.groupby(['VisitNumber'])['TripType'].first().values
    train.drop('TripType', axis=1, inplace=True)
    vn_train = train['VisitNumber'].unique()
    vn_test = test['VisitNumber'].unique()
    data_all = train.append(test).sort_values(by=['VisitNumber'], ascending=True)
    data_all, _ = aggregate_features(data_all)
    X_train = data_all[vn_train-1]
    X_test = data_all[vn_test-1]

    return X_train, y, X_test







