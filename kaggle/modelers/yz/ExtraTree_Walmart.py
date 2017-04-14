# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 09:58:10 2017

@author: Yunsong Zhang
"""

from time import time
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.model_selection import GridSearchCV
#import Pickle as pickle
import json

# Load cleaned data from training set
f = open('Train_data.json','r')
train_set_data = np.asarray(json.load(f))
f.close()
X = train_set_data[:,1:-1]

LE_Class = preprocessing.LabelEncoder()
Y        = LE_Class.fit_transform(train_set_data[:,-1])



n_jobs=8
print("Fitting ExtraTreesClassifier on data with %d cores..." % n_jobs)
t0 = time()

forest = ExtraTreesClassifier(n_estimators=300,
                              max_features=38,
                              n_jobs=n_jobs,
                              random_state=0,
                              max_depth=10)

forest.fit(X,Y)
print("done in %0.3fs" % (time() - t0))

f = open('Test_data.json','r')
test_set_data = np.asarray(json.load(f))
f.close()
Xts = test_set_data[:,1:]

Yts = forest.predict(Xts)
Yprob = forest.predict_proba(Xts)
decoded_pred = LE_Class.inverse_transform(Yts.astype(int))

f = open('ExtraTree_prob_pred.json','w')
json.dump(Yprob.tolist(),f)
f.close()

f = open('triptype_list.json','w')
json.dump(LE_Class.inverse_transform(np.arange(0,38)).tolist(),f)
f.close()

f = open('test_set_visit_number_list.json','w')
json.dump(test_set_data[:,0].tolist(),f)
f.close()


