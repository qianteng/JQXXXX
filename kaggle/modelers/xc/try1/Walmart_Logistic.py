import pandas as pd
import numpy as np
import scipy as sp
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train[train.FinelineNumber.notnull()]
train_part = train[:]
test_part = test[:]

model = LogisticRegression()
x = train_part[['Weekday', 'DepartmentDescription']]
y = train_part[['TripType']]
x = pd.get_dummies(x)

z = test_part[['Weekday', 'DepartmentDescription']]
zend = pd.DataFrame({'DepartmentDescription': ['HEALTH AND BEAUTY AIDS']},index = [len(z)])
z = z.append(zend)
z = pd.get_dummies(z)

model.fit(x, y)

submission = model.predict_proba(z)
submissiondf = pd.DataFrame(submission)
submissiondf.drop(len(submissiondf)-1)

index = test.iloc[:,0]
pre_out = pd.concat([index,submissiondf], axis = 1)
out = pre_out.groupby(pre_out.VisitNumber).mean()
out.reset_index(drop = True, inplace = True)
out.columns = ['VisitNumber', 'TripType_3','TripType_4','TripType_5','TripType_6','TripType_7',\
'TripType_8','TripType_9','TripType_12','TripType_14','TripType_15','TripType_18',\
'TripType_19','TripType_20','TripType_21','TripType_22','TripType_23','TripType_24',\
'TripType_25','TripType_26','TripType_27','TripType_28','TripType_29','TripType_30',\
'TripType_31','TripType_32','TripType_33','TripType_34','TripType_35','TripType_36',\
'TripType_37','TripType_38','TripType_39','TripType_40','TripType_41','TripType_42',\
'TripType_43','TripType_44','TripType_999']

out[['VisitNumber']] = out[['VisitNumber']].astype(int)
out.to_csv('Walmart_Logistic.csv', index = False)
