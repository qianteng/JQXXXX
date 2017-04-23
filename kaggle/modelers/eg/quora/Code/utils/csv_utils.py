# -*- coding: utf-8 -*-
"""
@author: Eric Guo <guoanjie@gmail.com>
@brief: utils for csv

"""

import pandas as pd


def _save(fname, df, encoding='utf-8'):
	df.to_csv(fname, encoding=encoding)

def _load(fname):
	return pd.read_csv(fname)
