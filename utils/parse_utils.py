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
config.read(base + '/../config/conf.cf')

def get_token(key):
	return int(float(config['VOCAB INFO'][key]))

zero_mask = get_token('zero_mask')
unk_token = get_token('unk_token')
start_token = get_token('start_token')
end_token = get_token('end_token')
div_token = get_token('div_token')
special_dict = {zero_mask: '<Z>', unk_token: '<UNK>', start_token: '<S>', end_token: '</S>', div_token: '<||>'}

#Interprets vocabulary given a list of IDs and a vocab
def interp_vocab_from_tokens(tokens, interp):
	return [interp[token] if token in interp else special_dict[token] for token in tokens]

#Encodes tokens given a vocabulary
def encode_tokens(tokens, vocab):
	return [vocab[token] if token in vocab else unk_token for token in tokens]

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
		print(read_file)
		line_num = 0
		while line:
			json_obj = json.loads(line)
			#Note: accessor should return None if something fails.
			for i in range(len(write_filepaths)):
				accessor = json_to_string_accessors[i]
				string = accessor(json_obj)
				if string == None:
					continue
				add_vocab_from_string(string, vocabs[i], tokenize[i], char[i], include_special[i], lower[i])
			line = read_file.readline()
			if line_num % 10000 == 0:
				print(line_num)
			line_num += 1
	for i in range(len(write_filepaths)):
		vocab, interps = process_vocab(vocabs[i], min_cutoffs[i])
		with open(write_filepaths[i], 'wb+') as f:
			pickle.dump([vocab, interps], f)
