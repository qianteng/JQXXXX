import numpy as np
import scipy as sp
import pandas as pd
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from utility_common import feature_extraction, data_path, result_path, log_path
import ipdb
import matplotlib.pyplot as plt

def tune_num_round(num_round = 5):
    """Use cross validation to find the optimal num_round
       num_round=462 gives the minimum test-mlogloss-mean
    """
    #history = pd.DataFrame(history, columns=['test-mlogloss-mean',  'test-mlogloss-std',
    #                                         'train-mlogloss-mean', 'train-mlogloss-std'])    # the columns of history
    X1, target, v_train, v_test = feature_extraction(useUpc=True)
    y = pd.get_dummies(target).values.argmax(1)
    N = X1.shape[0]
    seed = 137
    xgb_params = {'objective':'multi:softprob', 'num_class':38,
                  'eta':.2, 'max_depth':5, 'colsample_bytree':.4, 'subsample':.8,
                  'silent':1, 'eval_metric':'mlogloss', 'seed':seed}
    dtrain = xgb.DMatrix(X1[v_train-1], label = y)
    dtest = xgb.DMatrix(X1[v_test-1])
    history = xgb.cv(xgb_params, dtrain, num_round, nfold = 3, stratified = True, metrics = 'mlogloss',
                     verbose_eval = True, early_stopping_rounds = 50)
    np.save(log_path + 'num_round_tuning.npy', history)
    plt.errorbar(range(num_round), history['train-mlogloss-mean'],
                 history['train-mlogloss-std'], linestyle='None', marker='s', label='train', mfc=None, ms = 2)
    plt.errorbar(range(num_round), history['test-mlogloss-mean'],
                 history['test-mlogloss-std'], linestyle='None', marker='o', label='test', mfc=None, ms = 2)
    plt.legend()
    plt.xlabel('Num_round')
    plt.ylabel('mlogloss')
    plt.savefig(log_path + 'cv.eps', format='eps', dpi=1000)

def tune_params(params):
    """Use cross validation to find the optimal parameter
    
    Args:
         params: dict, a dict of parameters to tune, the values of the dict must be a list
    Returns:
         best_params: best parameters found by cross validation
         cv_result: detailed result of cross validation
    """
    
    X1, target, v_train, v_test = feature_extraction(useUpc=True)
    y = pd.get_dummies(target).values.argmax(1)
    N = X1.shape[0]  
    seed = 157
    xgb_params = {'learning_rate': [.2],
                  'n_estimators': [3],
                  'gamma': [0], 'max_depth': [5], 'min_child_weight': [1],
                  'subsample': [1], 'colsample_bytree': [.4], 'colsample_bylevel': [.8],
                  'reg_alpha': [0], 'reg_lambda': [1]}
    xgb_params.update(params)
    clf = xgb.XGBClassifier(silent = True, objective = 'multi:softprob', seed = seed)
    bst = GridSearchCV(clf, xgb_params, scoring = 'neg_log_loss',
                       cv = 3, refit = False).fit(X1[v_train-1], y)      # don't specify n_jobs in GridSearchCV. It seems that
                                                                         # lauching multiple xgb will make xgb crash. xgb has built-in multi-processing already
    best_params = bst.best_params_
    cv_result = bst.cv_results_
    return best_params, cv_result

if __name__ == '__main__':
    #tune_num_round()
    
    params = {'n_estimators': [2,3]}
    best_params, cv_result = tune_params(params)
    
    
    
 

