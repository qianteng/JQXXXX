# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@author: Eric Guo <guoanjie@gmail.com>
@brief: process data
"""

import regex

import config
from utils import logging_utils, pkl_utils, time_utils


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
	one -> 2
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
	def __init__self():
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


#----------------------- Processor Wrapper -----------------------
class ProcessorWrapper(object):
	def __init__(self, processor):
		self.processor = processor

	def transform(self, input):
		if isinstance(input, str) or isinstance(input, unicode):
			out = self.processor.transform(input)
		elif isinstance(input, float) or isinstance(input, int):
			out = self.processor.transform(str(input))
		elif isinstance(input, list):
			# take care when the input is a list
			out = [0] * len(input)
			for i in range(len(input)):
				out[i] = ProcessorWrapper(self.processor).transform(input[i])
		else:
			raise(ValueError("Currently not support type: {}".format(type(input).__name__)))
		return out


#------------------- List/DataFrame Processor Wrapper -------------------
class ListProcessor(object):
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


class DataFrameProcessor(object):
	"""
	WARNING: This class will operate on the original input dataframe itself
	"""
	def __init__(self, processors):
		self.processors = processors

	def process(self, df):
		for processor in self.processors:
			df = df.apply(ProcessorWrapper(processor).transform)
		return df


class DataFrameParallelProcessor(object):
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


#-------------------------- Main --------------------------
now = time_utils._timestamp()

def main():

	###########
	## Setup ##
	###########
	logname = "data_process_%s.log"%now
	logger = logging_utils._get_logger(config.LOG_DIR, logname)

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
		UnitConverter(),
		LowerUpperCaseSplitter(),
		# WordReplacer(replace_fname=config.WORD_REPLACER_DATA),
		LetterLetterSplitter(),
		DigitLetterSplitter(),
		DigitCommaDigitMerger(),
		NumberDigitMapper(),
	]

	## simple tests
	text = [
		"What would a Trump presidency mean for current international master’s students on an F1 visa?",
		"When will the Pokémon series end?",
		"Emoticons: What does “:/” mean?",
		"What will be the impact of scrapping of ₹500 and ₹1000 rupee notes on the real estate market?",
		"Why does Quora mark my questions as needing improvement/clarification before I have time to give it details? Literally within seconds…",
		"जिस स्थान का आपने भ्रमण किया है उसपर 50-60 शब्दों में प्रतिवेदन लिखिए?",
		"How long will it take to heat 750kg of water by 10°C with a 2060W heater?",
		"What does ℝ² mean?",
		r"How can I calculate the value of [math]\displaystyle\lim_{x\to ∞} \frac{5^{x+1}+7^{x+1}}{5^x-7^x}[/math] ?",
	]
	list_processor = ListProcessor(processors)
	processed = list_processor.process(text)
	for original, after in zip(text, processed):
		print
		print "Original:"
		print original
		print "After:"
		print after

	#############
	## Process ##
	#############
	## load raw data
	dfAll = pkl_utils._load(config.ALL_DATA_RAW)
	columns_to_proc = [col for col in columns_to_proc if col in dfAll.columns]

	## clean uisng a list of processors
	df_processor = DataFrameParallelProcessor(processors, config.DATA_PROCESSOR_N_JOBS)
	df_processor.process(dfAll, columns_to_proc)
	if config.TASK == "sample":
		print dfAll[columns_to_proc]
	# save data
	logger.info("Save to {}".format(config.ALL_DATA_LEMMATIZED))
	pkl_utils._save(config.ALL_DATA_LEMMATIZED, dfAll)


if __name__ == "__main__":
    main()
