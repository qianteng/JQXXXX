# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@author: Eric Guo <guoanjie@gmail.com>
@brief: splitter for Quora project

"""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import config
from utils import pkl_utils


class QuoraSplitter:
	def __init__(self, dfTrain, dfTest, n_iter=5, random_state=config.RANDOM_SEED, verbose=False):
		self.dfTrain = dfTrain
		self.dfTest = dfTest
		self.n_iter = n_iter
		self.random_state = random_state
		self.verbose = verbose

	def __str__(self):
		return "QuoraSplitter"

	def split(self):
		if self.verbose:
			print "*" * 50
			print "Naive Split"
		self.splits = [0] * self.n_iter
		tscv = TimeSeriesSplit(n_splits=self.n_iter)
		for run, (trainInd, validInd) in enumerate(tscv.split(self.dfTrain)):
			if self.verbose:
				print "=" * 50

			self.splits[run] = trainInd, validInd

			if self.verbose:
				print "-" * 50
				print "Index for run: {}".format(run + 1)
				print "Train (num = {})".format(len(trainInd))
				print trainInd
				print "Valid (num = {})".format(len(validInd))
				print validInd

		return self

	def save(self, fname):
		pkl_utils._save(fname, self.splits)


def main():

	dfTrain = pd.read_csv(config.TRAIN_DATA)
	dfTest = pd.read_csv(config.TEST_DATA)


	# splits for level1
	splitter = QuoraSplitter(dfTrain=dfTrain, dfTest=dfTest, n_iter=config.N_RUNS, random_state=config.RANDOM_SEED, verbose=True)
	splitter.split()
	splitter.save(config.SPLIT_DIR + "/splits_level1.pkl")


	## splits for level2
	splits_level1 = pkl_utils._load(config.SPLIT_DIR + "/splits_level1.pkl")
	splits_level2 = [0] * config.N_RUNS
	for run, (trainInd, validInd) in enumerate(splits_level1):
		dfValid = dfTrain.iloc[validInd].copy()
		splitter2 = QuoraSplitter(dfTrain=dfValid, dfTest=dfTest, n_iter=config.N_RUNS, random_state=run, verbose=True)
		splitter2.split()
		splits_level2[run] = splitter2.splits[-1]
		pkl_utils._save(config.SPLIT_DIR + "/splits_level2.pkl", splits_level2)


	## splits for level3
	splits_level2 = pkl_utils._load(config.SPLIT_DIR + "/splits_level2.pkl")
	splits_level3 = [0] * config.N_RUNS
	for run, (trainInd, validInd) in enumerate(splits_level2):
		dfValid = dfTrain.iloc[validInd].copy()
		splitter3 = QuoraSplitter(dfTrain=dfValid, dfTest=dfTest, n_iter=config.N_RUNS, random_state=run, verbose=True)
		splitter3.split()
		splits_level3[run] = splitter3.splits[-1]
		pkl_utils._save(config.SPLIT_DIR + "/splits_level3.pkl", splits_level3)


if __name__ == "__main__":
    main()
