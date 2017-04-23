#!/bin/bash

# @author: Eric Guo <guoanjie@gmail.com>
# @brief: generate all the data and features in one shot
# @note: if you don't have access to multi-core computers, drop the "&" in the cmd

#-----------------------------------------------------------------------
# prepare data
python data_prepare.py
