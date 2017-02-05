import os
import operator
import json
import pickle
import sys
base = os.path.dirname(__file__)
sys.path.append(base)
import configparser
import string_utils

config = configparser.ConfigParser()

#Adds vocabulary to the vocab_to_add_to given a list of tokens.
def add_vocab_from_tokens(tokens, vocab_to_add_to):
	for token in tokens:
		if not token:
			continue
		if token in vocab_to_add_to:
			vocab_to_add_to[token] += 1
		else:
			vocab_to_add_to[token] = 1

#Adds vocab present in a string based on a few options.
def add_vocab_from_string(string, vocab_to_add_to, tokenize = True, char = True, include_special = True, lower = True):
	if char:
		tokens = string_utils.parse_string_to_chars(string, lower = lower, include_special = include_special)
		add_vocab_from_tokens(tokens, vocab_to_add_to)
	else:
		tokens = string_utils.parse_string_to_words(string, lower = lower, include_special = include_special)
		add_vocab_from_tokens(tokens, vocab_to_add_to)

def process_vocab(vocab, min_cutoff):
	sorted_vocab_list = sorted(vocab.items(), key = operator.itemgetter(1), reverse = True)
	vocab_trimmed_inds = {}
	for i in range(len(sorted_vocab_list)):
		if sorted_vocab_list[i][1] > min_cutoff:
			vocab_trimmed_inds[sorted_vocab_list[i][0]] = 5 + i
	vocab_trimmed_interps = {}
	for key in vocab_trimmed_inds:
		vocab_trimmed_interps[vocab_trimmed_inds[key]] = key
	return vocab_trimmed_inds, vocab_trimmed_interps
	
def parse_vocab(filepaths, write_filepaths, json_to_string_accessors, tokenize = None, char = None,  lower = None, include_special = None, min_cutoffs = None):
	assert len(write_filepaths) == len(json_to_string_accessors), 'Number of vocabulary items is not the same as number of json accessors'
	
	if not tokenize:
		tokenize = [True for _ in range(len(write_filepaths))]
	if not char:
		char = [True for _ in range(len(write_filepaths))]
	if not lower:
		lower = [True for _ in range(len(write_filepaths))]
	if not include_special:
		include_special = [True for _ in range(len(write_filepaths))]	
	if not min_cutoffs:
		min_cutoffs = [0 for _ in range(len(write_filepaths))]

	vocabs = [{} for _ in range(len(write_filepaths))]
	
	for filepath in filepaths:
		read_file = open(filepath, 'r')
		line = read_file.readline()
		linenum = 0
		while line:
			json_obj = json.loads(line)
			for i in range(len(write_filepaths)):
				accessor = json_to_string_accessors[i]
				string = accessor(json_obj)
				add_vocab_from_string(string, vocabs[i], tokenize[i], char[i], include_special[i], lower[i])
			line = read_file.readline()
			if linenum % 5000 == 0:
				print('Processing vocab line ' + str(linenum))
			linenum += 1
	
	for i in range(len(write_filepaths)):
		vocab, interps = process_vocab(vocabs[i], min_cutoffs[i])
		with open(write_filepaths[i], 'wb+') as f:
			pickle.dump([vocab, interps], f)
