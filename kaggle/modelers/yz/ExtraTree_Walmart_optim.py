# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:41:25 2017

@author: Yunsong Zhang
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 09:58:10 2017

@author: Yunsong Zhang
"""

from time import time
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
#import Pickle as pickle
import json

# Load cleaned data from training set
f = open('Train_data.json','r')
train_set_data = np.asarray(json.load(f))
f.close()
X = train_set_data[:,1:-1]

LE_Class = preprocessing.LabelEncoder()
Y        = LE_Class.fit_transform(train_set_data[:,-1])


param_dict = {}
param_dict['n_jobs'] = 8
param_dict['max_features'] = 38


t0 = time()

model = ExtraTreesClassifier(**param_dict)

clf = GridSearchCV(model,{'n_estimators':[300,400,500,600,700],
                          'max_depth':[2,4,6,8,10]},verbose=1)

clf.fit(X,Y)


#Read Test set data
f = open('Test_data.json','r')
test_set_data = np.asarray(json.load(f))
f.close()
Xts = test_set_data[:,1:]

print(clf.best_score_)
print(clf.best_params_)




Yts = clf.predict(Xts)
Yprob = clf.predict_proba(Xts)
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
