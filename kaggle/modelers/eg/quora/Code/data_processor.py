# -*- coding: utf-8 -*-
"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@author: Eric Guo <guoanjie@gmail.com>
@brief: process data
"""

from pprint import pprint

import config
from utils import logging_utils, time_utils


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
	]

	## simple tests
	text = [
		"What would a Trump presidency mean for current international master’s students on an F1 visa?",
		"When will the Pokémon series end?",
		"Emoticons: What does “:/” mean?",
		"What will be the impact of scrapping of ₹500 and ₹1000 rupee notes on the real estate market?",
		"Why does Quora mark my questions as needing improvement/clarification before I have time to give it details? Literally within seconds…",
		"When travelling to a new region is it better to immerse yourself in 1–2 cities or to see as many cities as you can cram in?",
		"जिस स्थान का आपने भ्रमण किया है उसपर 50-60 शब्दों में प्रतिवेदन लिखिए?",
		"How long will it take to heat 750kg of water by 10°C with a 2060W heater?",
		"How list showing the month and a number for each month . ☝January 713 ☝February 823 ☝March 531 ☝ April 542 ☝May 351 ☝June 462 ☝July 471 ☝ August 683 ⚡Decode the logic and find the number for September = ? iska answer dena?",
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


if __name__ == "__main__":
    main()
