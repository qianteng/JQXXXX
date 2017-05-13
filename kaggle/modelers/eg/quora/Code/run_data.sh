#!/bin/bash

# @author: Eric Guo <guoanjie@gmail.com>
# @brief: generate all the data and features in one shot
# @note: if you don't have access to multi-core computers, drop the "&" in the cmd

#-----------------------------------------------------------------------
# prepare data
python data_preparer.py

#-----------------------------------------------------------------------
# process/clean data
python data_processor.py


#-----------------------------------------------------------------------
# generate basic features
python feature_basic.py &


#-----------------------------------------------------------------------
# generate distance features
python feature_distance.py jaccard &

python feature_distance.py edit &


#-----------------------------------------------------------------------
# generate first and last ngram features
python feature_first_last_ngram.py &
