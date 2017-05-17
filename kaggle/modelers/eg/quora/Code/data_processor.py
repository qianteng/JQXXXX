# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: process data
        - a bunck of processing
        - automated spelling correction
        - query expansion
        - extract product name for search_term and product_title

"""

import csv
import imp

import nltk
import regex
import numpy as np
import pandas as pd
import multiprocessing
from bs4 import BeautifulSoup
from collections import Counter

import config
from utils import ngram_utils, pkl_utils, logging_utils, time_utils


#--------------------------- Processor ---------------------------
## base class
## Most of the processings can be casted into the "pattern-replace" framework
class BaseReplacer(object):
    def __init__(self, pattern_replace_pair_list=[]):
        self.pattern_replace_pair_list = pattern_replace_pair_list
    def transform(self, text):
        for pattern, replace in self.pattern_replace_pair_list:
            try:
                text = regex.sub(pattern, replace, text)
            except:
                pass
        return regex.sub(r"\s+", " ", text).strip()


class UnicodeConverter(BaseReplacer):
    def transform(self, text):
        return unicode(text, 'utf-8')


## deal with case
class LowerCaseConverter(BaseReplacer):
    """
    Traditional -> traditional
    """
    def transform(self, text):
        return text.lower()


class LowerUpperCaseSplitter(BaseReplacer):
    """
    homeBASICS Traditional Real Wood -> homeBASICS Traditional Real Wood

    hidden from viewDurable rich finishLimited lifetime warrantyEncapsulated panels ->
    hidden from view Durable rich finish limited lifetime warranty Encapsulated panels

    Dickies quality has been built into every product.Excellent visibilityDurable ->
    Dickies quality has been built into every product Excellent visibility Durable

    BAD CASE:
    shadeMature height: 36 in. - 48 in.Mature width
    minutesCovers up to 120 sq. ft.Cleans up
    PUT one UnitConverter before LowerUpperCaseSplitter

    Reference:
    https://www.kaggle.com/c/home-depot-product-search-relevance/forums/t/18472/typos-in-the-product-descriptions
    """
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"(\w)[\.?!]([A-Z])", r"\1 \2"),
            (r"(?<=( ))([a-z]+)([A-Z]+)", r"\2 \3"),
        ]


## deal with word replacement
# 1st solution in CrowdFlower
class WordReplacer(BaseReplacer):
    def __init__(self, replace_fname):
        self.replace_fname = replace_fname
        self.pattern_replace_pair_list = []
        for line in csv.reader(open(self.replace_fname)):
            if len(line) == 1 and line[0].startswith("#"):
                continue
            try:
                pattern = r"(?<=\W|^)%s(?=\W|$)"%line[0]
                replace = line[1]
                self.pattern_replace_pair_list.append( (pattern, replace) )
            except:
                print(line)
                pass


## deal with letters
class LetterLetterSplitter(BaseReplacer):
    """
    For letter and letter
    /:
    Cleaner/Conditioner -> Cleaner Conditioner

    -:
    Vinyl-Leather-Rubber -> Vinyl Leather Rubber

    For digit and digit, we keep it as we will generate some features via math operations,
    such as approximate height/width/area etc.
    /:
    3/4 -> 3/4

    -:
    1-1/4 -> 1-1/4
    """
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"([a-zA-Z]+)[/\-]([a-zA-Z]+)", r"\1 \2"),
        ]


## deal with digits and numbers
class DigitLetterSplitter(BaseReplacer):
    """
    x:
    1x1x1x1x1 -> 1 x 1 x 1 x 1 x 1
    19.875x31.5x1 -> 19.875 x 31.5 x 1

    -:
    1-Gang -> 1 Gang
    48-Light -> 48 Light

    .:
    includes a tile flange to further simplify installation.60 in. L x 36 in. W x 20 in. ->
    includes a tile flange to further simplify installation. 60 in. L x 36 in. W x 20 in.
    """
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"(\d+)[\.\-]*([a-zA-Z]+)", r"\1 \2"),
            (r"([a-zA-Z]+)[\.\-]*(\d+)", r"\1 \2"),
        ]


class DigitCommaDigitMerger(BaseReplacer):
    """
    1,000,000 -> 1000000
    """
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"(?<=\d+),(?=000)", r""),
        ]


class NumberDigitMapper(BaseReplacer):
    """
    one -> 1
    two -> 2
    """
    def __init__(self):
        numbers = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
        ]
        digits = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90
        ]
        self.pattern_replace_pair_list = [
            (r"(?<=\W|^)%s(?=\W|$)"%n, str(d)) for n,d in zip(numbers, digits)
        ]


## deal with unit
class UnitConverter(BaseReplacer):
    """
    shadeMature height: 36 in. - 48 in.Mature width
    PUT one UnitConverter before LowerUpperCaseSplitter
    """
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"([0-9]+)( *)(inches|inch|in|in.|')\.?", r"\1 in. "),
            (r"([0-9]+)( *)(pounds|pound|lbs|lb|lb.)\.?", r"\1 lb. "),
            (r"([0-9]+)( *)(foot|feet|ft|ft.|'')\.?", r"\1 ft. "),
            (r"([0-9]+)( *)(square|sq|sq.) ?\.?(inches|inch|in|in.|')\.?", r"\1 sq.in. "),
            (r"([0-9]+)( *)(square|sq|sq.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 sq.ft. "),
            (r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(inches|inch|in|in.|')\.?", r"\1 cu.in. "),
            (r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 cu.ft. "),
            (r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1 gal. "),
            (r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1 oz. "),
            (r"([0-9]+)( *)(centimeters|cm)\.?", r"\1 cm. "),
            (r"([0-9]+)( *)(milimeters|mm)\.?", r"\1 mm. "),
            (r"([0-9]+)( *)(minutes|minute)\.?", r"\1 min. "),
            (r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1 deg. "),
            (r"([0-9]+)( *)(v|volts|volt)\.?", r"\1 volt. "),
            (r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1 watt. "),
            (r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp. "),
            (r"([0-9]+)( *)(qquart|quart)\.?", r"\1 qt. "),
            (r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1 hr "),
            (r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1 gal. per min. "),
            (r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gal. per hr "),
        ]


## deal with html tags
class HtmlCleaner:
    def __init__(self, parser):
        self.parser = parser

    def transform(self, text):
        bs = BeautifulSoup(text, self.parser)
        text = bs.get_text(separator=" ")
        return text


## deal with some special characters
# 3rd solution in CrowdFlower (cleanData_02.R)  
class QuartetCleaner(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"<.+?>", r""),
            # html codes
            (r"&nbsp;", r" "),
            (r"&amp;", r"&"),
            (r"&#39;", r"'"),
            (r"/>/Agt/>", r""),
            (r"</a<gt/", r""),
            (r"gt/>", r""),
            (r"/>", r""),
            (r"<br", r""),
            # do not remove [".", "/", "-", "%"] as they are useful in numbers, e.g., 1.97, 1-1/2, 10%, etc.
            (r"[ &<>)(_,;:!?\+^~@#\$]+", r" "),
            ("'s\\b", r""),
            (r"[']+", r""),
            (r"[\"]+", r""),
        ]


## lemmatizing for using pretrained word2vec model
# 2nd solution in CrowdFlower
class Lemmatizer:
    def __init__(self):
        self.Tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.Lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    def transform(self, text):
        tokens = [self.Lemmatizer.lemmatize(token) for token in self.Tokenizer.tokenize(text)]
        return " ".join(tokens)


## stemming
class Stemmer:
    def __init__(self, stemmer_type="snowball"):
        self.stemmer_type = stemmer_type
        if self.stemmer_type == "porter":
            self.stemmer = nltk.stem.PorterStemmer()
        elif self.stemmer_type == "snowball":
            self.stemmer = nltk.stem.SnowballStemmer("english")

    def transform(self, text):
        tokens = [self.stemmer.stem(token) for token in text.split(" ")]
        return " ".join(tokens)


#----------------------- Processor Wrapper -----------------------
class ProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor

    def transform(self, input):
        if isinstance(input, str) or isinstance(input, unicode):
            out = self.processor.transform(input)
        elif isinstance(input, float) or isinstance(input, int):
            out = self.processor.transform(str(input))
        elif isinstance(input, list):
            # take care when the input is a list
            # currently for a list of attributes
            out = [0]*len(input)
            for i in range(len(input)):
                out[i] = ProcessorWrapper(self.processor).transform(input[i])
        else:
            raise(ValueError("Currently not support type: %s"%type(input).__name__))
        return out


#------------------- List/DataFrame Processor Wrapper -------------------
class ListProcessor:
    """
    WARNING: This class will operate on the original input list itself
    """
    def __init__(self, processors):
        self.processors = processors

    def process(self, lst):
        for i in range(len(lst)):
            for processor in self.processors:
                lst[i] = ProcessorWrapper(processor).transform(lst[i])
        return lst


class DataFrameProcessor:
    """
    WARNING: This class will operate on the original input dataframe itself
    """
    def __init__(self, processors):
        self.processors = processors

    def process(self, df):
        for processor in self.processors:
            df = df.apply(ProcessorWrapper(processor).transform)
        return df


class DataFrameParallelProcessor:
    """
    WARNING: This class will operate on the original input dataframe itself

    https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap
    """
    def __init__(self, processors, n_jobs=4):
        self.processors = processors
        self.n_jobs = n_jobs

    def process(self, dfAll, columns):
        df_processor = DataFrameProcessor(self.processors)
        for col in columns:
            dfAll[col] = df_processor.process(dfAll[col])
        return dfAll


#------------------- Query Expansion -------------------
# 3rd solution in CrowdFlower
class QueryExpansion:
    def __init__(self, df, ngram=3, stopwords_threshold=0.9, base_stopwords=set()):
        self.df = df[["search_term", "product_title"]].copy()
        self.ngram = ngram
        self.stopwords_threshold = stopwords_threshold
        self.stopwords = set(base_stopwords).union(self._get_customized_stopwords())
        
    def _get_customized_stopwords(self):
        words = " ".join(list(self.df["product_title"].values)).split(" ")
        counter = Counter(words)
        num_uniq = len(list(counter.keys()))
        num_stop = int((1.-self.stopwords_threshold)*num_uniq)
        stopwords = set()
        for e,(w,c) in enumerate(sorted(counter.items(), key=lambda x: x[1])):
            if e == num_stop:
                break
            stopwords.add(w)
        return stopwords

    def _ngram(self, text):
        tokens = text.split(" ")
        tokens = [token for token in tokens if token not in self.stopwords]
        return ngram_utils._ngrams(tokens, self.ngram, " ")

    def _get_alternative_query(self, df):
        res = []
        for v in df:
            res += v
        c = Counter(res)
        value, count = c.most_common()[0]
        return value

    def build(self):
        self.df["title_ngram"] = self.df["product_title"].apply(self._ngram)
        corpus = self.df.groupby("search_term").apply(lambda df: self._get_alternative_query(df["title_ngram"]))
        corpus = corpus.reset_index()
        corpus.columns = ["search_term", "search_term_alt"]
        self.df = pd.merge(self.df, corpus, on="search_term", how="left")
        return self.df["search_term_alt"].values


#------------------- Process Attributes -------------------
def _split_attr_to_text(text):
    attrs = text.split(config.ATTR_SEPARATOR)
    return " ".join(attrs)

def _split_attr_to_list(text):
    attrs = text.split(config.ATTR_SEPARATOR)        
    if len(attrs) == 1:
        # missing
        return [[attrs[0], attrs[0]]]
    else:
        return [[n,v] for n,v in zip(attrs[::2], attrs[1::2])]


#-------------------------- Main --------------------------
now = time_utils._timestamp()

def main():

    ###########
    ## Setup ##
    ###########
    logname = "data_processor_%s.log"%now
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    # put product_attribute_list, product_attribute and product_description first as they are
    # quite time consuming to process
    columns_to_proc = [
        "question1",
        "question2",
    ]
    if config.PLATFORM == "Linux":
        config.DATA_PROCESSOR_N_JOBS = len(columns_to_proc)

    # clean using a list of processors
    processors = [
        UnicodeConverter(),
        LowerCaseConverter(), 
        # See LowerUpperCaseSplitter and UnitConverter for why we put UnitConverter here
        UnitConverter(),
        LowerUpperCaseSplitter(), 
        # WordReplacer(replace_fname=config.WORD_REPLACER_DATA), 
        LetterLetterSplitter(),
        DigitLetterSplitter(), 
        DigitCommaDigitMerger(), 
        NumberDigitMapper(),
        UnitConverter(), 
        QuartetCleaner(), 
        HtmlCleaner(parser="html.parser"), 
        Lemmatizer(),
    ]
    stemmers = [
        Stemmer(stemmer_type="snowball"), 
        Stemmer(stemmer_type="porter")
    ][0:1]

    ## simple test
    text = "1/2 inch rubber lep tips Bullet07"
    print("Original:")
    print(text)
    list_processor = ListProcessor(processors)
    print("After:")
    print(list_processor.process([text]))

    #############
    ## Process ##
    #############
    ## load raw data
    dfAll = pkl_utils._load(config.ALL_DATA_RAW)
    columns_to_proc = [col for col in columns_to_proc if col in dfAll.columns]


    ## clean uisng a list of processors
    df_processor = DataFrameParallelProcessor(processors, config.DATA_PROCESSOR_N_JOBS)
    df_processor.process(dfAll, columns_to_proc)
    # save data
    logger.info("Save to %s"%config.ALL_DATA_LEMMATIZED)
    pkl_utils._save(config.ALL_DATA_LEMMATIZED, dfAll)


    ## clean using stemmers
    df_processor = DataFrameParallelProcessor(stemmers, config.DATA_PROCESSOR_N_JOBS)
    df_processor.process(dfAll, columns_to_proc)
    # save data
    logger.info("Save to %s"%config.ALL_DATA_LEMMATIZED_STEMMED)
    pkl_utils._save(config.ALL_DATA_LEMMATIZED_STEMMED, dfAll)


if __name__ == "__main__":
    main()
    logging_utils._sms()
