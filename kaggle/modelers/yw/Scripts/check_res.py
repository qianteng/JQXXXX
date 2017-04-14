import numpy as np
import pandas as pd
import feature_engineering as fe


def pred_to_submit_version(pred_file_name):
    pred = pd.read_pickle(fe.get_file_path(pred_file_name))
    pred = pd.DataFrame(pred)
    inds = pd.read_csv(fe.get_file_path('test.csv'), usecols=['VisitNumber'])
    pred.index = inds['VisitNumber'].tolist()

    TripType_map = pd.read_csv(fe.get_file_path('TripType_map.csv'), header=None).set_index([0])
    pred.columns = TripType_map.ix[pred.columns]

    submit_df = pd.concat([pred.idxmax(axis=1), pred.max(axis=1)], axis=1).reset_index()
    submit_df.columns = ['VisitNumber', 'TripType', 'prob']
    submit_pivot = submit_df.groupby('VisitNumber').max().pivot(columns='TripType')

    col_names = np.unique(submit_df['TripType']).flatten().tolist()
    col_names = {cc[0]:'TripType_%d'%cc[0] for cc in col_names}

    submit_pivot = submit_pivot.rename(columns=col_names)['prob']
    submit_pivot.columns.name = None

    submit_temp = pd.read_csv(fe.get_file_path('sample_submission.csv'), nrows=0, index_col='VisitNumber')
    submit_pivot = submit_temp.append(submit_pivot)

    submit_pivot = submit_pivot.divide(submit_pivot).fillna(0)
    submit_pivot.to_csv(fe.get_file_path('submit_pivot_{}.csv'.format(pred_file_name)))


    return submit_pivot

res = pred_to_submit_version('pred_101_nopca')
res.shape
if(0):
    res = pred_to_submit_version('pred_200')
    res = pred_to_submit_version('pred_10')

    sample_df = pd.read_csv(fe.get_file_path('sample_submission.csv'), index_col='VisitNumber')
    submit_pivot_200 = pd.read_csv(fe.get_file_path('submit_pivot_200.csv'))

    sample_df.shape
    submit_pivot_10.shape
    submit_pivot_200.shape

    sample_df.columns
    submit_pivot_200.columns