import numpy as np
import scipy as sp
import pandas as pd
from datetime import datetime
import xgboost as xgb
import sklearn
from sklearn.metrics import log_loss
from utility_common import feature_extraction, data_path



# Returns: all the cols of X used by xgboost
def feature_selection(X, y, xgb_params, num_round):
    dtrain = xgb.DMatrix(X, label = y)
    bst = xgb.train(xgb_params, dtrain, num_round)
    imp = bst.get_fscore() #  {'f1':123, 'f12':344, 'f15':131, ..}
    keys = imp.keys()
    cols = np.sort([int(k[1:]) for k in keys])
    return cols

if __name__ == '__main__':
    X1, target, v_train, v_test = feature_extraction(useUpc=True)
    X = X1[v_train-1]
    y = pd.get_dummies(target).values.argmax(1)
    num_round = 5
    xgb_params = {'objective':'multi:softprob', 'num_class':38,
                  'eta':.2, 'max_depth':5, 'colsample_bytree':.4, 'subsample':.8,
                  'silent':1, 'nthread':8}
    cols = feature_selection(X, y, xgb_params, num_round)
