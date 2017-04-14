import numpy as np
import pandas as pd

from utility_common import data_path, file_train, file_test, result_path, log_path

training = pd.read_csv(file_train)
test = pd.read_csv(file_test)
target = training.groupby('VisitNumber').TripType.first().values
v_train = training.VisitNumber.unique()
v_test = test.VisitNumber.unique()

pr_xgb = np.load(result_path + 'pr002_xgb_test.npy')

type_str_lst = ['TripType_'+str(c) for c in np.unique(target)]
pred = pr_xgb

pred002 = pd.DataFrame({'VisitNumber':v_test}).join(pd.DataFrame(pred, columns=type_str_lst))
pred002.to_csv(result_path+'pred002.csv', index = False)
