
# coding: utf-8

# In[ ]:

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import calendar
import numpy as np
import pandas as pd
import xgboost as xgb


# In[ ]:

train = pd.read_csv('~/Kaggle/walmart/train.csv', dtype={'Upc': str, 'FinelineNumber': str})


# In[ ]:

test = pd.read_csv('~/Kaggle/walmart/test.csv', dtype={'Upc': str, 'FinelineNumber': str})


# In[ ]:

target_column = train.columns[0]
target_column


# In[ ]:

data = train.append(test)[train.columns]


# In[ ]:

feature_columns = []


# In[ ]:

data['NumberItems'] = data.groupby('VisitNumber')['VisitNumber'].transform('count')
feature_columns += ['NumberItems']


# In[ ]:

day_number_by_name = {d: n for n, d in enumerate(calendar.day_name)}
data['WeekdayNumber'] = data['Weekday'].map(day_number_by_name)
data['IsWeekday'] = data['WeekdayNumber'] < 5
feature_columns += ['WeekdayNumber', 'IsWeekday']


# In[ ]:

# data['LenUpc'] = data['Upc'].fillna('').apply(len)
# feature_columns += ['LenUpc']

data['Upc'] = data['Upc'].fillna('0').apply(int)
feature_columns += ['Upc']


# In[ ]:

data['TotalScanCount'] = data.groupby('VisitNumber')['ScanCount'].transform('sum')
feature_columns += ['ScanCount', 'TotalScanCount']


# In[ ]:

departments = list(data['DepartmentDescription'].unique())
feature_columns += departments

for department in departments:
    data[department] = data['DepartmentDescription'] == department


# In[ ]:

# data['LenFinelineNumber'] = data['FinelineNumber'].fillna('').apply(len)
# feature_columns += ['LenFinelineNumber']

data['FinelineNumber'] = data['FinelineNumber'].fillna('0').apply(int)
feature_columns += ['FinelineNumber']


# In[ ]:

feature_columns


# In[ ]:

data.head()


# In[ ]:

data.tail()


# In[ ]:

train_mask = data[target_column].notnull()
X_train = data[train_mask][feature_columns].as_matrix()
y_train = data[train_mask][target_column].as_matrix()
X_test = data[~train_mask][feature_columns].as_matrix()
sample_weight = 1 / data['NumberItems']


# In[ ]:

gbm = xgb.XGBClassifier().fit(X_train, y_train, sample_weight=sample_weight)


# In[ ]:

predictions = gbm.predict_proba(X_test)


# In[ ]:

columns = ['TripType_{}'.format(int(t)) for t in gbm._le.inverse_transform(range(predictions.shape[1]))]


# In[ ]:

submission = pd.concat([test['VisitNumber'], pd.DataFrame(predictions, columns=columns)], axis=1)


# In[ ]:

submission = submission.groupby('VisitNumber').mean()


# In[ ]:

submission.to_csv('submission.csv')

