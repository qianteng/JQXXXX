# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: utils for time

"""

import datetime
import os

from subprocess import Popen, PIPE


def _timestamp():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H-%M")
    return now_str


def _timestamp_pretty():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M")
    return now_str


def _timestamp_logging():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return now_str

def _etime():
    pid = str(os.getpid())
    (stdout, stderr) = Popen(["ps", "-p", pid, "-o", "etime="], stdout=PIPE).communicate()
    return stdout
