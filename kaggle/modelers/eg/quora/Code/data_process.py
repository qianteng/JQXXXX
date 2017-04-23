# -*- coding: utf-8 -*-
"""
@author: Eric Guo <guoanjie@gmail.com>
@author: Chenglong Chen <c.chenglong@gmail.com>
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


if __name__ == "__main__":
    main()
