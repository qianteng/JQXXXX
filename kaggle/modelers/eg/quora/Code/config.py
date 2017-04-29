# -*- coding: utf-8 -*-
"""
@author: Eric Guo <guoanjie@gmail.com>
@brief: config for Quora project
"""

import multiprocessing
import platform
import subprocess

from utils import os_utils


# ---------------------- Overall -----------------------
TASK = 'all'
# for testing data processing and feature generation
TASK = "sample"
SAMPLE_SIZE = 1000

# ------------------------ PATH ------------------------
ROOT_DIR = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).rstrip() + '/kaggle/modelers/eg/quora'

DATA_DIR = ROOT_DIR + "/Data"
CLEAN_DATA_DIR = DATA_DIR + "/Clean"

LOG_DIR = ROOT_DIR + "/Log"

# ------------------------ DATA ------------------------
# provided data
TRAIN_DATA = DATA_DIR + "/train.csv"
TEST_DATA = DATA_DIR + "/test.csv"

ALL_DATA_RAW = CLEAN_DATA_DIR + "/all.csv.pkl"
ALL_DATA_UTF8 = CLEAN_DATA_DIR + "/all.utf-8.csv.pkl"

# size
TRAIN_SIZE = 404290
if TASK == "sample":
    TRAIN_SIZE = SAMPLE_SIZE
TEST_SIZE = 2345796


# ------------------------ OTHER ------------------------
PLATFORM = platform.system()
NUM_CORES = multiprocessing.cpu_count()

DATA_PROCESSOR_N_JOBS = NUM_CORES
AUTO_SPELLING_CHECKER_N_JOBS = NUM_CORES


# ---------------------- CREATE PATH --------------------
DIRS = []
DIRS += [CLEAN_DATA_DIR]
DIRS += [LOG_DIR]

os_utils._create_dirs(DIRS)
