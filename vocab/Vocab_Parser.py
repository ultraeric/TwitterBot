import json
import sys
import os
import re
import pickle
import operator
import sys
import configparser
from utils import string_utils as suts
from utils import parse_utils as puts

base = os.path.dirname(__file__)

config = configparser.ConfigParser()
config.read(base + '/../config/conf.cf')

def parse_vocab():

	clean_dataset_dir = base + '/.' + config['FILE LOCS']['clean_dataset_dir']
	filepaths = [clean_dataset_dir + '/' + filep for filep in os.listdir(clean_dataset_dir) if filep.endswith('.txt')]
	write_vocab_base = base + '/.' + config['FILE LOCS']['vocab_dir']
	write_filepaths = [write_vocab_base + '/vocab.data']
	lower = [False]
	json_to_string_accessors = [lambda json_obj: json_obj['text']]
	min_cutoffs = [int(float(config['DATA INFO']['min_cutoffs']))]
	puts.parse_vocab(filepaths, write_filepaths, json_to_string_accessors, lower = lower, min_cutoffs = min_cutoffs)	
