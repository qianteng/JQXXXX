import numpy as np
import scipy as sp
import pandas as pd
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import log_loss
from utility_common import feature_extraction, data_path, result_path, log_path
import ipdb
import matplotlib.pyplot as plt



if __name__ == '__main__':

    X1, target, v_train, v_test = feature_extraction(useUpc=True)
    y = pd.get_dummies(target).values.argmax(1)
    N = X1.shape[0]
    num_round = 600
    seed = 112
    xgb_params = {'objective':'multi:softprob', 'num_class':38,
                  'eta':.2, 'max_depth':5, 'colsample_bytree':.4, 'subsample':.8,
                  'silent':1,
                  'seed':seed, 'eval_metric':'mlogloss'}
    dtrain = xgb.DMatrix(X1[v_train-1], label = y)
    dtest = xgb.DMatrix(X1[v_test-1])
    """
    t0 = datetime.now()
    watchlist = [(dtrain, 'train')]
    evals_result = {}
    bst = xgb.train(xgb_params, dtrain, num_round, watchlist, evals_result = evals_result)
    pr_xgb_test = bst.predict(dtest)
    pr_xgb_train = bst.predict(dtrain)
    print('Time used: {}'.format(datetime.now() - t0))
    plt.plot(evals_result['train']['mlogloss'])
    plt.show()
    """

    # use cross validation to find the optimal num_round
    history = xgb.cv(xgb_params, dtrain, num_round, nfold = 3, stratified = True, metrics = 'mlogloss', verbose_eval = True)
    np.save(log_path + 'num_round_tuning.npy', history)
    plt.errorbar(range(num_round), history['train-mlogloss-mean'],
                 history['train-mlogloss-std'], linestyle='None', marker='s', label='train', mfc=None, ms = 2)
    plt.errorbar(range(num_round), history['test-mlogloss-mean'],
                 history['test-mlogloss-std'], linestyle='None', marker='o', label='test', mfc=None, ms = 2)
    plt.legend()
    plt.xlabel('Num_round')
    plt.ylabel('mlogloss')
    plt.show()
    
    
