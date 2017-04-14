import pandas as pd
from scipy import stats
import numpy as np
import scipy as sp


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
    df = df.join(groupby_vn.agg({'Encoded_Upc' : pd.Series.nunique}), on='VisitNumber', rsuffix='_nunique_groupby_vn')
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
    out['Encoded_UPC_nunique'] = groupby_vn.agg({'Encoded_Upc': pd.Series.nunique})
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
    data = df.groupby(['VisitNumber', 'Encoded_Upc'], as_index=False)['ScanCount'].count()
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

    return out


# Convert categoricla variable into dummy variables and return a dense matrix
def generate_categorical_features(df, row_name, col_name, val_name, aggregate=False, quantile=0.1):

    row_labels = df[row_name].unique()
    label2row = pd.Series(range(row_labels.size), index=row_labels)

    if aggregate:
        df = df.groupby([row_name, col_name], as_index=False)[val_name].sum()

    col_stat = df.groupby(col_name, as_index=False)[val_name].sum()
    col_labels = col_stat[col_stat[val_name] > col_stat[val_name].quantile(q=quantile)][col_name]

    #col_labels = df[col_name].unique()
    label2col = pd.Series(range(col_labels.size), index=col_labels)

    cols = df[col_name].map(label2col)
    valid_mask = ~cols.isnull()
    rows = df[row_name].map(label2row)[valid_mask]
    vals = df[val_name][valid_mask]
    cols = cols[valid_mask]

    dense_matrix = sp.sparse.coo_matrix((vals, (rows, cols)), shape=(label2row.size, label2col.size)).tocsr()
    return dense_matrix


def aggregate_features(df):
    y = df.groupby(['VisitNumber'])['TripType'].first().values
    general_features = generate_general_features(df)
    general_features.drop('dep_has_max', axis=1, inplace=True)
    fineline_features = generate_categorical_features(df, 'VisitNumber', 'FinelineNumber', 'ScanCount', aggregate=True,
                                                      quantile=0.1)
    department_features = generate_categorical_features(df, 'VisitNumber', 'DepartmentDescription', 'ScanCount',
                                                        aggregate=True, quantile=0.0)

    upc_features = generate_categorical_features(df, 'VisitNumber', 'Encoded_Upc', 'ScanCount', aggregate=True,
                                                quantile=0.6)

    X = sp.sparse.hstack((general_features, fineline_features, department_features, upc_features)).tocsr()
    return X, y










