#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:18:16 2017

@author: yunsongzhang
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import cPickle as pickle
import json




"""
   Step 1, data cleaning and preprocessing
"""

# Data read and exploration
train_df = pd.read_csv('./data/train.csv')
test_df  = pd.read_csv('./data/test.csv')

for C_name in train_df.columns:
    if np.any( train_df[C_name].isnull() ):
        print('In train.csv, NaN exists in %s\n',C_name)

for C_name in test_df.columns:
    if np.any( test_df[C_name].isnull() ):
        print('In test.csv, NaN exists in %s\n',C_name)

# Fill missing data in the data frame
train_df['Upc']                   = train_df['Upc'].fillna(-1)
test_df['Upc']                    = test_df['Upc'].fillna(-1)     
train_df['DepartmentDescription'] = train_df['DepartmentDescription'].fillna('UNKNOWN')
test_df['DepartmentDescription']  = test_df['DepartmentDescription'].fillna('UNKNOWN')
train_df['FinelineNumber']        = train_df['FinelineNumber'].fillna(-1)
test_df['FinelineNumber']         = test_df['FinelineNumber'].fillna(-1)


# Replace nominal input into numbers
LE_Weekday = preprocessing.LabelEncoder()
train_encoded_weekday = LE_Weekday.fit_transform(train_df['Weekday'])
test_encoded_weekday  = LE_Weekday.transform(test_df['Weekday'])
train_df['Weekday']   = train_encoded_weekday
test_df['Weekday']    = test_encoded_weekday


LE_Depart = preprocessing.LabelEncoder()
train_encoded_depart = LE_Depart.fit_transform(train_df['DepartmentDescription'])
test_encoded_depart  = LE_Depart.transform(test_df['DepartmentDescription'])
train_df['DepartmentDescription'] = train_encoded_depart
test_df['DepartmentDescription']  = test_encoded_depart


# Convert the dataframe into ndarray and get X_train,Y_train & X_test

# train set 
train_basic_vis_info = train_df[['VisitNumber','Weekday','TripType']].drop_duplicates().sort('VisitNumber')
mat_train_basic_vis_info   = train_basic_vis_info.as_matrix()

train_depart_info    = train_df[['VisitNumber','DepartmentDescription','ScanCount']]
grouped_train        = train_depart_info.groupby(['VisitNumber','DepartmentDescription'],as_index=False).sum()

LE_VN_tr   = preprocessing.LabelEncoder()
grouped_train['VisitNumber'] = LE_VN_tr.fit_transform(grouped_train['VisitNumber'])

num_Visits      = len(grouped_train['VisitNumber'].unique())
num_Departments = len(grouped_train['DepartmentDescription'].unique())
print('There are %d departments in total %d visits\n' %(num_Departments,num_Visits))

mat_train_depart = np.zeros([num_Visits,num_Departments],dtype=int)

def record_train_data(x):
    mat_train_depart[x[0],x[1]] = x[2]
    
grouped_train.apply(record_train_data,axis=1)

Data_train_set = np.concatenate((mat_train_basic_vis_info[:,0:2],mat_train_depart,mat_train_basic_vis_info[:,2:3]),axis=1)

    










# test set
test_basic_vis_info = test_df[['VisitNumber','Weekday']].drop_duplicates().sort('VisitNumber')
mat_test_basic_vis_info   = test_basic_vis_info.as_matrix()

test_depart_info    = test_df[['VisitNumber','DepartmentDescription','ScanCount']]
grouped_test        = test_depart_info.groupby(['VisitNumber','DepartmentDescription'],as_index=False).sum()

LE_VN_ts   = preprocessing.LabelEncoder()
grouped_test['VisitNumber'] = LE_VN_ts.fit_transform(grouped_test['VisitNumber'])

num_Visits      = len(grouped_test['VisitNumber'].unique())
num_Departments_ts = len(grouped_test['DepartmentDescription'].unique())
print('There are %d departments in total %d visits \n' %(num_Departments_ts,num_Visits))

mat_test_depart = np.zeros([num_Visits,num_Departments],dtype=int)

def record_test_data(x):
    mat_test_depart[x[0],x[1]] = x[2]
    
grouped_test.apply(record_test_data,axis=1)

Data_test_set = np.concatenate((mat_test_basic_vis_info[:,0:2],mat_test_depart),axis=1)


f = open('Data_Train','wb')
pickle.dump(Data_train_set,f)
f.close()

f = open('Data_Test','wb')
pickle.dump(Data_test_set,f)
f.close()

f = open('Train_data.json','w')
json.dump(Data_train_set.tolist(),f)
f.close()

f = open('Test_data.json','w')
json.dump(Data_test_set.tolist(),f)
f.close()


# Dump the cleaned data into Pickle and json forms
