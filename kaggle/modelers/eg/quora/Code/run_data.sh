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


#-----------------------------------------------------------------------
# generate intersect features
python feature_intersect_count.py &

python feature_intersect_position.py &


#-----------------------------------------------------------------------
# generate match features
python feature_match.py &


#-----------------------------------------------------------------------
# generate statistical cooccurrence (weighted) features
python feature_stat_cooc_tfidf.py tf &

python feature_stat_cooc_tfidf.py tfidf &

python feature_stat_cooc_tfidf.py bm25 &
