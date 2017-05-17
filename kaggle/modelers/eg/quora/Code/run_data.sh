#!/bin/bash

# @author: Eric Guo <guoanjie@gmail.com>
# @brief: generate all the data and features in one shot
# @note: if you don't have access to multi-core computers, drop the "&" in the cmd

pids=()

#-----------------------------------------------------------------------
# prepare data
python data_preparer.py

#-----------------------------------------------------------------------
# process/clean data
python data_processor.py


#-----------------------------------------------------------------------
# generate basic features
python feature_basic.py &
pids+=($!)


#-----------------------------------------------------------------------
# generate distance features
python feature_distance.py jaccard &
pids+=($!)
python feature_distance.py edit &
pids+=($!)


#-----------------------------------------------------------------------
# generate first and last ngram features
python feature_first_last_ngram.py &
pids+=($!)


#-----------------------------------------------------------------------
# generate intersect features
python feature_intersect_count.py &
pids+=($!)
python feature_intersect_position.py &
pids+=($!)


#-----------------------------------------------------------------------
# generate match features
python feature_match.py &
pids+=($!)


#-----------------------------------------------------------------------
# generate statistical cooccurrence (weighted) features
python feature_stat_cooc_tfidf.py tf &
pids+=($!)
python feature_stat_cooc_tfidf.py tfidf &
pids+=($!)
python feature_stat_cooc_tfidf.py bm25 &
pids+=($!)


#-----------------------------------------------------------------------
# generate word2vec features using pre-trained word2vec model
python feature_word2vec.py google &
pids+=($!)
python feature_word2vec.py wikipedia &
pids+=($!)


#-----------------------------------------------------------------------
# generate wordnet similarity features
# time consuming part ~20 hrs
python feature_wordnet_similarity.py &
pids+=($!)


#-----------------------------------------------------------------------
# generate vector space features
# most memory consuming part > 16GB
# python feature_vector_space.py


if [ $USER == "ubuntu" ]
	then
		for pid in ${pids[*]}
		do
			wait $pid
		done
		sudo poweroff
fi
