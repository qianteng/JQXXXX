# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:21:56 2017

@author: Yunsong Zhang
"""


import numpy as np
import pandas as pd
import json

f  = open('XGBoost_prob_pred_200.json','r')
Yprob  = np.array(json.load(f))
f.close()

f = open('triptype_list.json','r')
typenames = json.load(f)
f.close()

colnames =['VisitNumber']
for num in typenames:
    name_str = 'TripType_'+str(num)
    colnames.append(name_str)

f = open('test_set_visit_number_list.json','r')
visit_numbers = np.asarray(json.load(f)).reshape((-1,1))
f.close()

#Vis = pd.Series(visit_numbers.astype(int32))
Data2submit = np.concatenate((visit_numbers,Yprob),axis=1)

submission = pd.DataFrame(Data2submit,columns=colnames)
submission['VisitNumber'] = submission['VisitNumber'].astype(np.int32)
                          
submission.to_csv('submission_XGBoost_0d3.csv',index=False)

