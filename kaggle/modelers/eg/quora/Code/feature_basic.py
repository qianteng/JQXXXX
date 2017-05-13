# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@author: Eric Guo <guoanjie@gmail.com>
@brief: basic features

"""

import config
from utils import time_utils, logging_utils, pkl_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper


class DocId(BaseEstimator):
	def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
		super(DocId, self).__init__(obs_corpus, target_corpus, aggregation_mode)
		obs_set = set(obs_corpus)
		self.encoder = dict(zip(obs_set, range(len(obs_set))))

	def __name__(self):
		return "DocId"

	def transform_one(self, obs, target, id):
		return self.encoder[obs]


#---------------- Main ---------------------------
def main():
	logname = "generate_feature_basic_%s.log"%time_utils._timestamp()
	logger = logging_utils._get_logger(config.LOG_DIR, logname)
	dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)

	## basic
	generators = [DocId]
	obs_fields = ["question1", "question2"]
	for generator in generators:
		param_list = []
		sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger)
		sf.go()


if __name__ == "__main__":
	main()
