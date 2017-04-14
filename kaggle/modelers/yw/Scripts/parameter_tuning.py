import os
import time

import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from feature_engineering import get_big_df_train, data_dir, big_df_pca_transform, pickle_dump


def gsearch_1(XX_train, YY_train, gparam, out_path):
    """
    grid search for: max_depth, min_child_weight    
    """

    clf_parms = {'learning_rate': 0.1, 'n_estimators': 3, \
                 'objective': 'multi:softprob', 'gamma': 0, 'subsample': 0.8, \
                 'colsample_bytree': 0.8, 'seed': 27}

    param_test1 = gparam
    gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(clf_parms, silent=False),
     param_grid = param_test1, scoring='accuracy', verbose=1, n_jobs=4,cv=3)
    gsearch1.fit(XX_train,YY_train)
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

    res = gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    pickle_dump(res, out_path)

    return

def gsearch_2(XX_train, YY_train, gparam, out_path):
    """
        grid search for: gamma
    """
    clf_parms = {'learning_rate': 0.1, 'n_estimators': 3, \
                 'objective': 'multi:softprob', 'gamma': 0, 'subsample': 0.8, \
                 'colsample_bytree': 0.8, 'seed': 27, 'min_child_weight': 1, 'num_class': 38}

    param_test2 = gparam

    gsearch1 = GridSearchCV(estimator=xgb.XGBClassifier(clf_parms, silent=False),
                            param_grid=param_test2, scoring='accuracy', verbose=1, n_jobs=4, cv=3)
    gsearch1.fit(XX_train, YY_train)
    res = gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    pickle_dump(res, out_path)

    return

def gsearch_3(XX_train, YY_train, gparam, out_path):
    """
        grid search for: subsample, colsample_bytree
    """
    clf_parms = {'learning_rate': 0.1, 'n_estimators': 1, \
                 'objective': 'multi:softprob', 'gamma': 0, 'subsample': 0.8, \
                 'colsample_bytree': 0.8, 'seed': 27, 'max_depth':9, 'min_child_weight': 1, 'num_class': 38}

    param_test3 = gparam

    gsearch3 = GridSearchCV(estimator=xgb.XGBClassifier(clf_parms, silent=False),
                            param_grid=param_test3, scoring='accuracy', verbose=1, n_jobs=4, cv=3)
    gsearch3.fit(XX_train, YY_train)
    res = gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
    pickle_dump(res, out_path)

    return



if __name__ == '__main__':

    init_time = time.clock()
    curr_time = init_time
    print('start ... ')

    # load train data
    big_df_train = get_big_df_train()
    YY_train = big_df_train['newTripType']
    XX_train = big_df_train[[col for col in big_df_train.columns if col not in ['newTripType']]]

    # pca tranform
    print('PCA ... ', time.clock() - curr_time)
    curr_time = time.clock()

    XX_train_pca = big_df_pca_transform(XX_train)
    dtrain = xgb.DMatrix(XX_train_pca, YY_train)


    # grid search 1: max_depth, min_child_weight
    if(0):
        print('GridSearch1 start... ', time.clock() - curr_time)
        curr_time = time.clock()
        if (0):
            gparam = {'max_depth':range(3,10,2),
                        'min_child_weight':range(1,10,2)}
            out_path = os.path.join(data_dir, 'gsearch1_full_primary')
        if (0):
            gparam = {'max_depth': [1, 3, 5, 9, 11, 13, 15, 20, 25],
                        'min_child_weight': [1, 5, 9, 15]}

            out_path = os.path.join(data_dir, 'gsearch1_full_further')

        gsearch_1(XX_train_pca, YY_train, gparam, out_path)

        print('GridSearch1 finish... ', time.clock() - curr_time)
        curr_time = time.clock()

    # grid search 2: gamma
    if (0):
        print('GridSearch2 start... ', time.clock() - curr_time)
        curr_time = time.clock()

        gparam = {'gamma': [i / 10.0 for i in range(0, 5)]}
        out_path = os.path.join(data_dir, 'gsearch2_full')
        gsearch_2(XX_train_pca, YY_train, gparam, out_path)

        print('GridSearch2 finish... ', time.clock() - curr_time)
        curr_time = time.clock()

    # grid search 3: subsample, colsample_bytree
    if (0):
        print('GridSearch3 start... ', time.clock() - curr_time)
        curr_time = time.clock()

        gparam = {
                    'subsample': [i / 10.0 for i in range(6, 10)],
                    'colsample_bytree': [i / 10.0 for i in range(6, 10)]
                }
        out_path = os.path.join(data_dir, 'gsearch3_full')
        gsearch_3(XX_train_pca, YY_train, gparam, out_path)

        print('GridSearch3 finish... ', time.clock() - curr_time)
        curr_time = time.clock()

    print('finish .... ', time.clock()-init_time)