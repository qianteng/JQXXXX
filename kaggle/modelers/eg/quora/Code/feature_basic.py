# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@author: Eric Guo <guoanjie@gmail.com>
@brief: basic features

"""

import re
from collections import Counter

import numpy as np

import config
from utils import ngram_utils, nlp_utils, np_utils
from utils import time_utils, logging_utils, pkl_utils
from feature_base import BaseEstimator, StandaloneFeatureWrapper


# tune the token pattern to get a better correlation with y_train
# token_pattern = r"(?u)\b\w\w+\b"
# token_pattern = r"\w{1,}"
# token_pattern = r"\w+"
# token_pattern = r"[\w']+"
token_pattern = " " # just split the text into tokens


class DocId(BaseEstimator):
	def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
		super(DocId, self).__init__(obs_corpus, target_corpus, aggregation_mode)
		obs_set = set(obs_corpus)
		self.encoder = dict(zip(obs_set, range(len(obs_set))))

	def __name__(self):
		return "DocId"

	def transform_one(self, obs, target, id):
		return self.encoder[obs]


class DocLen(BaseEstimator):
	"""Length of document"""
	def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
		super(DocLen, self).__init__(obs_corpus, target_corpus, aggregation_mode)

	def __name__(self):
		return "DocLen"

	def transform_one(self, obs, target, id):
		obs_tokens = nlp_utils._tokenize(obs, token_pattern)
		return len(obs_tokens)


class DocFreq(BaseEstimator):
	"""Frequency of the document in the corpus"""
	def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
		super(DocFreq, self).__init__(obs_corpus, target_corpus, aggregation_mode)
		self.counter = Counter(obs_corpus)

	def __name__(self):
		return "DocFreq"

	def transform_one(self, obs, target, id):
		return self.counter[obs]


class DocEntropy(BaseEstimator):
	"""Entropy of the document"""
	def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
		super(DocEntropy, self).__init__(obs_corpus, target_corpus, aggregation_mode)

	def __name__(self):
		return "DocEntropy"

	def transform_one(self, obs, target, id):
		obs_tokens = nlp_utils._tokenize(obs, token_pattern)
		counter = Counter(obs_tokens)
		count = np.asarray(list(counter.values()))
		proba = count/np.sum(count).astype(float)
		return np_utils._entropy(proba)


class DigitCount(BaseEstimator):
	"""Count of digit in the document"""
	def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
		super(DigitCount, self).__init__(obs_corpus, target_corpus, aggregation_mode)

	def __name__(self):
		return "DigitCount"

	def transform_one(self, obs, target, id):
		return len(re.findall(r"\d", obs))


class DigitRatio(BaseEstimator):
	def __init__(self, obs_corpus, target_corpus, aggregation_mode=""):
		super(DigitRatio, self).__init__(obs_corpus, target_corpus, aggregation_mode)

	def __name__(self):
		return "DigitRatio"

	def transform_one(self, obs, target, id):
		obs_tokens = nlp_utils._tokenize(obs, token_pattern)
		return np_utils._try_divide(len(re.findall(r"\d", obs)), len(obs_tokens))


class UniqueCount_Ngram(BaseEstimator):
	def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
		super(UniqueCount_Ngram, self).__init__(obs_corpus, target_corpus, aggregation_mode)
		self.ngram = ngram
		self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

	def __name__(self):
		return "UniqueCount_{}".format(self.ngram_str)

	def transform_one(self, obs, target, id):
		obs_tokens = nlp_utils._tokenize(obs, token_pattern)
		obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
		return len(set(obs_ngrams))


class UniqueRatio_Ngram(BaseEstimator):
	def __init__(self, obs_corpus, target_corpus, ngram, aggregation_mode=""):
		super(UniqueRatio_Ngram, self).__init__(obs_corpus, target_corpus, aggregation_mode)
		self.ngram = ngram
		self.ngram_str = ngram_utils._ngram_str_map[self.ngram]

	def __name__(self):
		return "UniqueRatio_{}".format(self.ngram_str)

	def transform_one(self, obs, target, id):
		obs_tokens = nlp_utils._tokenize(obs, token_pattern)
		obs_ngrams = ngram_utils._ngrams(obs_tokens, self.ngram)
		return np_utils._try_divide(len(set(obs_ngrams)), len(obs_ngrams))


#---------------- Main ---------------------------
def main():
	logname = "generate_feature_basic_%s.log"%time_utils._timestamp()
	logger = logging_utils._get_logger(config.LOG_DIR, logname)
	dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)

	## basic
	generators = [DocId, DocLen, DocFreq, DocEntropy, DigitCount, DigitRatio]
	obs_fields = ["question1", "question2"]
	for generator in generators:
		param_list = []
		sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger)
		sf.go()

	## unique count
	generators = [UniqueCount_Ngram, UniqueRatio_Ngram]
	obs_fields = ["question1", "question2"]
	ngrams = [1,2,3]
	for generator in generators:
		for ngram in ngrams:
			param_list = [ngram]
			sf = StandaloneFeatureWrapper(generator, dfAll, obs_fields, param_list, config.FEAT_DIR, logger)
			sf.go()


if __name__ == "__main__":
	main()
