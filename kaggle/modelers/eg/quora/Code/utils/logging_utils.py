# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for logging

"""

import os
import logging
import logging.handlers

from . import os_utils, sms_utils, time_utils


def _get_logger(logdir, logname, loglevel=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt)

    handler = logging.handlers.RotatingFileHandler(
                    filename=os.path.join(logdir, logname),
                    maxBytes=10*1024*1024, 
                    backupCount=10)
    handler.setFormatter(formatter)

    logger = logging.getLogger("")
    logger.addHandler(handler)
    logger.setLevel(loglevel)
    return logger

def _succeeded():
    sms_utils._send("[{}]\n报告主人，程序运行完啦！୧(๑•̀⌄•́๑)૭✧\n{}".format(time_utils._timestamp_logging(), os_utils._command()))
