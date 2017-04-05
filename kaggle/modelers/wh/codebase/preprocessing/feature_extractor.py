import pandas as pd
from scipy import stats
import numpy as np


def aggregate_features(df):
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
            print "%d of %d fineline number features have been added." %(count, len(fn_list))
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
