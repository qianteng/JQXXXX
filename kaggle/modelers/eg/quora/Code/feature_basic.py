# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@author: Eric Guo <guoanjie@gmail.com>
@brief: basic features

"""

import config
from utils import time_utils, logging_utils, pkl_utils


#---------------- Main ---------------------------
def main():
	logname = "generate_feature_basic_%s.log"%time_utils._timestamp()
	logger = logging_utils._get_logger(config.LOG_DIR, logname)
	dfAll = pkl_utils._load(config.ALL_DATA_LEMMATIZED_STEMMED)


if __name__ == "__main__":
	main()
