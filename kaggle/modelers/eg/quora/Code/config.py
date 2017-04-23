# -*- coding: utf-8 -*-
"""
@author: Eric Guo <guoanjie@gmail.com>
@brief: config for Quora project
"""

import subprocess


# ---------------------- Overall -----------------------
TASK = 'all'
# # for testing data processing and feature generation
# TASK = "sample"
SAMPLE_SIZE = 1000

# ------------------------ PATH ------------------------
ROOT_DIR = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).rstrip() + '/kaggle/modelers/eg/quora'

DATA_DIR = ROOT_DIR + '/Data'
