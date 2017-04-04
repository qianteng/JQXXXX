#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 22:11:45 2017

@author: yunsongzhang
"""

import numpy as np
import xgboost as xgb
from sklearn import preprocessing
import json


# Load cleaned data from training set 
f=open('Train_data.json','r')
train_set_data = np.asarray(json.load(f))
n = train_set_data.shape[0]

X = train_set_data[:,1:-1]
LE_Class = preprocessing.LabelEncoder()
Y        = LE_Class.fit_transform(train_set_data[:,-1])


train_X  = X[:int(n*0.7),:]
train_Y  = Y[:int(n*0.7)]

test_X   = X[int(n*0.7):,:]
test_Y   = Y[int(n*0.7):]


xg_train = xgb.DMatrix( train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.3
param['max_depth'] = 8
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 38
param['eval_metric'] = 'mlogloss'

watchlist = [ (xg_train,'train'), (xg_test, 'eval') ]
num_round = 500

evals_result ={}
bst = xgb.train(param, xg_train, num_round, evals=watchlist ,early_stopping_rounds=3,evals_result = evals_result)

#param_set = [6]
#for value in param_set:
#    param['max_depth'] = value
#    evals_result = {}
#    bst = xgb.train(param, xg_train, num_round, evals=watchlist ,early_stopping_rounds=3,evals_result = evals_result)
#    best_score = evals_result['eval']['mlogloss'][bst.best_iteration]
#    print('max_depth: %d, best_score: %f' %(value,best_score))
    
    

f = open('Test_data.json','r')
test_set_data = np.asarray(json.load(f))
f.close()
pred_X        = test_set_data[:,1:]
xg_pred       = xgb.DMatrix(pred_X)
pred_Yprob        = bst.predict(xg_pred).reshape((pred_X.shape[0],38))

#decoded_Y     = LE_Class.inverse_transform(pred_Y.astype(int))

f  = open('XGBoost_prob_pred_200.json','w')
json.dump(pred_Yprob.tolist(),f)
f.close()


f = open('triptype_list.json','w')
json.dump(LE_Class.inverse_transform(np.arange(0,38)).tolist(),f)
f.close()

f = open('test_set_visit_number_list.json','w')
json.dump(test_set_data[:,0].tolist(),f)
f.close()
#pickle.dump(bst,open('model200.pkl','wb'))
