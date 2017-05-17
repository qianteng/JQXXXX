# -*- coding: utf-8 -*-
"""
@author: Eric Guo <guoanjie@gmail.com>
@brief: generate raw dataframe data

"""

import gc

import numpy as np
import pandas as pd

import config
from utils import pkl_utils
from utils import logging_utils

def main():
	# load provided data
	dfTrain = pd.read_csv(config.TRAIN_DATA)
	dfTest  = pd.read_csv(config.TEST_DATA)

	#
	dfTest["is_duplicate"] = np.zeros(config.TEST_SIZE)

	# concat train and test
	dfAll = pd.concat([dfTrain, dfTest], ignore_index=True).fillna("")
	del dfTrain
	del dfTest
	gc.collect()

	# save data
	if config.TASK == "sample":
		dfAll = dfAll.iloc[:config.SAMPLE_SIZE].copy()
	pkl_utils._save(config.ALL_DATA_RAW, dfAll)


if __name__ == "__main__":
	main()
	logging_utils._sms()
