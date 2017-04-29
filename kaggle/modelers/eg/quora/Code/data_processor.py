# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@author: Eric Guo <guoanjie@gmail.com>
@brief: process data
"""

import config
from utils import logging_utils, time_utils


#----------------------- Processor Wrapper -----------------------
class ProcessorWrapper(object):
	def __init__(self, processor):
		self.processor = processor

	def transform(self, input):
		if isinstance(input, str) or isinstance(input, unicode):
			out = self.processor.transform(input)
		elif isinstance(input, float) or isinstance(input, int):
			out = self.processor.transform(str(input))
		elif isinstance(input, list):
			# take care when the input is a list
			out = [0] * len(input)
			for i in range(len(input)):
				out[i] = ProcessorWrapper(self.processor).transform(input[i])
		else:
			raise(ValueError("Currently not support type: {}".format(type(input).__name__)))
		return out


#------------------- List/DataFrame Processor Wrapper -------------------
class ListProcessor(object):
	"""
	WARNING: This class will operate on the original input list itself
	"""
	def __init__(self, processors):
		self.processors = processors

	def process(self, lst):
		for i in range(len(lst)):
			for processor in self.processors:
				lst[i] = ProcessorWrapper(processor).transform(lst[i])
		return lst


#-------------------------- Main --------------------------
now = time_utils._timestamp()

def main():

	###########
	## Setup ##
	###########
	logname = "data_process_%s.log"%now
	logger = logging_utils._get_logger(config.LOG_DIR, logname)

	columns_to_proc = [
		"question1",
		"question2",
	]
	if config.PLATFORM == "Linux":
		config.DATA_PROCESSOR_N_JOBS = len(columns_to_proc)

	# clean using a list of processors
	processors = [
	]


if __name__ == "__main__":
    main()
