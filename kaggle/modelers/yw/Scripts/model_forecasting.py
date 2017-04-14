import pickle
import pandas as pd
import time
import xgboost as xgb

import feature_engineering as fe

def model_forecasting(dtrain, dtest, clf_parms):

    init_time = time.clock()
    curr_time = init_time
    print('training...')

    bst = xgb.train(clf_parms, dtrain, num_boost_round=500)

    print('forecasting...', time.clock()-curr_time)
    curr_time = time.clock()
    pred = bst.predict(dtest)

    print('saving results...', time.clock()-curr_time)
    fimportance = bst.get_fscore()
    fimportance = [[score, feature] for (feature, score) in fimportance.items()]
    fimportance = pd.DataFrame(fimportance, columns=['score', 'feature']).set_index('feature').sort_values('score', ascending=False)

    fileobj = open(fe.get_file_path('pred'), 'wb')
    pickle.dump(pred, fileobj)
    fileobj.close()

    fimportance.to_csv(fe.get_file_path('fimportance.csv'))

    print('finishing...', time.clock()-init_time)

def pred_to_submit_version():
    pred = pd.read_pickle(fe.get_file_path('pred'))
    pred = pd.DataFrame(pred)
    inds = pd.read_csv(fe.get_file_path('test.csv'), usecols=['VisitNumber'])
    pred.index = inds['VisitNumber'].tolist()

    TripType_map = pd.read_csv(fe.get_file_path('TripType_map.csv'), header=None).set_index([0])
    pred.columns = TripType_map.ix[pred.columns]

    submit_df = pd.concat([pred.idxmax(axis=1), pred.max(axis=1)], axis=1).reset_index()
    submit_df.columns = ['VisitNumber', 'TripType', 'prob']
    submit_pivot = submit_df.groupby('VisitNumber').max().pivot(columns='TripType')
    submit_pivot = submit_pivot.divide(submit_pivot).fillna(0)

    submit_pivot.to_csv(fe.get_file_path('submit_pivot.csv'))
    return submit_pivot


res = pred_to_submit_version()


if __name__ == '__main__':

    init_time = time.clock()
    print('start...')

    # data preparation
    big_df_train, big_df_test = fe.get_big_df_train(), fe.get_big_df_test()
    big_df_train_pca, big_df_test_pca = fe.big_df_pca_transform(big_df_train, big_df_test)

    XX_train = big_df_train[[col for col in big_df_train.columns if col not in ['newTripType']]]
    YY_train = big_df_train['newTripType']

    dtrain, dtest = xgb.DMatrix(big_df_train_pca, YY_train), xgb.DMatrix(big_df_test_pca)

    # forecasting
    clf_parms = {'learning_rate': 0.1, 'n_estimators': 500, \
                     'objective': 'multi:softprob', 'gamma': 0, 'subsample': 0.8, \
                     'colsample_bytree': 0.8, 'max_depth':9, 'min_child_weight': 9, 'num_class': 38, 'seed': 27, 'silent':False}

    model_forecasting(dtrain, dtest, clf_parms)

    print('finishing...', time.clock()-init_time)