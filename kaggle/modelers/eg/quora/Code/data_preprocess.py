from __future__ import print_function

import gc

import numpy as np
import pandas as pd

import config
from utils import csv_utils, pkl_utils

def main():
	# load provided data
	dfTrain = pd.read_csv(config.TRAIN_DATA)
	dfTest  = pd.read_csv(config.TEST_DATA)

	# 
	print("Train Mean: {:0.6f}".format(np.mean(dfTrain["is_duplicate"])))
	print("Train Var:  {:0.6f}".format(np.var(dfTrain["is_duplicate"])))

	#
	dfTest["is_duplicate"] = np.zeros(config.TEST_SIZE)

	# concat train and test
	dfAll = pd.concat([dfTrain, dfTest], ignore_index=True)
	del dfTrain
	del dfTest
	gc.collect()

	# convert to utf-8
	csv_utils._save(config.ALL_DATA_CSV, dfAll)
	dfAll = csv_utils._load(config.ALL_DATA_CSV)

	# save data
	if config.TASK == "sample":
		dfAll = dfAll.iloc[:config.SAMPLE_SIZE].copy()
	pkl_utils._save(config.ALL_DATA_UTF8, dfAll)


if __name__ == "__main__":
	main()
