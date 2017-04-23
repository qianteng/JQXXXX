# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@author: Eric Guo <guoanjie@gmail.com>
@brief: process data
"""

import config
from utils import logging_utils, time_utils


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
