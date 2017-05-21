# -*- coding: utf-8 -*-
"""
@author: Eric Guo <guoanjie@gmail.com>
@brief: config for Quora project
"""

import multiprocessing
import platform
import subprocess

import numpy as np

from utils import os_utils


# ---------------------- Overall -----------------------
TASK = 'all'
# # for testing data processing and feature generation
# TASK = "sample"
SAMPLE_SIZE = 1000

# ------------------------ PATH ------------------------
ROOT_DIR = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).rstrip() + '/kaggle/modelers/eg/quora'

DATA_DIR = ROOT_DIR + "/Data"
CLEAN_DATA_DIR = DATA_DIR + "/Clean"

FEAT_DIR = ROOT_DIR + "/Feat"
FEAT_FILE_SUFFIX = ".pkl"
FEAT_CONF_DIR = ROOT_DIR + "/Code/conf"

LOG_DIR = ROOT_DIR + "/Log"

# word2vec/doc2vec/glove
WORD2VEC_MODEL_DIR = DATA_DIR + "/word2vec"
GLOVE_WORD2VEC_MODEL_DIR = DATA_DIR + "/glove/gensim"
# DOC2VEC_MODEL_DIR = "%s/doc2vec"%DATA_DIR

# index split
SPLIT_DIR = DATA_DIR + "/split"

# ------------------------ DATA ------------------------
# provided data
TRAIN_DATA = DATA_DIR + "/train.csv"
TEST_DATA = DATA_DIR + "/test.csv"

ALL_DATA_RAW = CLEAN_DATA_DIR + "/all.csv.pkl"
ALL_DATA_LEMMATIZED = CLEAN_DATA_DIR + "/all.lemmatized.csv.pkl"
ALL_DATA_LEMMATIZED_STEMMED = CLEAN_DATA_DIR + "/all.lemmatized.stemmed.csv.pkl"

# size
TRAIN_SIZE = 404290
if TASK == "sample":
    TRAIN_SIZE = SAMPLE_SIZE
TEST_SIZE = 2345796


# ------------------------ PARAM ------------------------
# cv
N_RUNS = 5
# N_FOLDS = 1

# intersect count/match
STR_MATCH_THRESHOLD = 0.85

# bm25
BM25_K1 = 1.6
BM25_B = 0.75

# svd
SVD_DIM = 100
SVD_N_ITER = 5

# count transformer

COUNT_TRANSFORM = np.log1p

# missing value
# MISSING_VALUE_STRING = "MISSINGVALUE"
MISSING_VALUE_NUMERIC = -1.


# ------------------------ OTHER ------------------------
RANDOM_SEED = 2016
PLATFORM = platform.system()
NUM_CORES = multiprocessing.cpu_count()

DATA_PROCESSOR_N_JOBS = NUM_CORES
AUTO_SPELLING_CHECKER_N_JOBS = NUM_CORES


# ---------------------- CREATE PATH --------------------
DIRS = []
DIRS += [CLEAN_DATA_DIR]
DIRS += [SPLIT_DIR]
DIRS += [FEAT_DIR, FEAT_CONF_DIR]
DIRS += [FEAT_DIR + "/All"]
DIRS += [FEAT_DIR + "/Combine"]
DIRS += [LOG_DIR]

os_utils._create_dirs(DIRS)
