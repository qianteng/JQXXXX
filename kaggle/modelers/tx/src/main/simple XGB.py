from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os, sys
import calendar
import numpy as np
import pandas as pd
import xgboost as xgb

dir = "/Users/tianyixia/dev/Qishi_ML_adv/kaggle/walmart/data/"

def preprocessing():
	train = pd.read_csv(os.path.join(dir, 'train.csv'), dtype={'Upc': str, 'FinelineNumber': str})
	test = pd.read_csv(os.path.join(dir, "/test.csv"), dtype={'Upc': str, 'FinelineNumber': str})

	target_column = train.columns[0]
	data = train.append(test)[train.columns]
	feature_columns = []

	data['NumberItems'] = data.groupby('VisitNumber')['VisitNumber'].transform('count')
	feature_columns += ['NumberItems']


	day_number_by_name = {d: n for n, d in enumerate(calendar.day_name)}
	data['WeekdayNumber'] = data['Weekday'].map(day_number_by_name)
	data['IsWeekday'] = data['WeekdayNumber'] < 5
	feature_columns += ['WeekdayNumber', 'IsWeekday']

	data['LenUpc'] = data['Upc'].fillna('').apply(len)
	feature_columns += ['LenUpc']

	data['TotalScanCount'] = data.groupby('VisitNumber')['ScanCount'].transform('sum')
	feature_columns += ['ScanCount', 'TotalScanCount']

	departments = list(data['DepartmentDescription'].unique())
	feature_columns += departments

	for department in departments:
	    data[department] = data['DepartmentDescription'] == department

	data['LenFinelineNumber'] = data['FinelineNumber'].fillna('').apply(len)
	feature_columns += ['FinelineNumber']

	return data, feature_columns, target_column

def trainXGB(data, feature_columns, target_column):
	train_mask = data[target_column].notnull()
	X_train = data[train_mask][feature_columns].as_matrix()
	y_train = data[train_mask][target_column].as_matrix()
	X_test = data[~train_mask][feature_columns].as_matrix()
	sample_weight = 1 / data['NumberItems']

	gbm = xgb.XGBClassifier().fit(X_train, y_train, sample_weight=sample_weight)


	predictions = gbm.predict_proba(X_test)

	columns = ['TripType_{}'.format(int(t)) for t in gbm._le.inverse_transform(range(predictions.shape[1]))]

	submission = pd.concat([test['VisitNumber'], pd.DataFrame(predictions, columns=columns)], axis=1)

	submission.to_csv(os.path.join(dir, '/submission.csv'))

	return gbm, predictions

if __name__ == '__main__':
	trainXGB(preprocessing())