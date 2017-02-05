import os
import operator
import json
import pickle
import sys
base = os.path.dirname(__file__)
import configparser
from utils import string_utils as suts

config = configparser.ConfigParser()
config.read(base + '/../config/conf.cf')

def get_token(key):
	return int(float(config['VOCAB INFO'][key]))

unk_token = get_token('unk_token')
start_token = get_token('start_token')
end_token = get_token('end_token')
div_token = get_token('div_token')

vocab = pickle.load(open(base + '/.' + config['FILE LOCS']['vocab_dir'] + '/vocab.data', 'rb'))
vocab = vocab[0]

def process():
	write_file = config['FILE LOCS']['preprocessed_dataset']
	write_file = open(write_file, 'a+')
	
	clean_dataset_dir = base + '/.' + config['FILE LOCS']['clean_dataset_dir']	
	for filename in os.listdir(clean_dataset_dir):
		if not filename.endswith('.txt'):
			continue
		read_file = open(clean_dataset_dir + '/' + filename, 'r')
		line = read_file.readline()
		line_num = 0
		while line:
			line = line.encode().decode()
			line.replace('\\n', '')
			chars = suts.parse_string_to_chars(line, lower = False)
			char_inds = [vocab[key] if key in vocab else unk_token for key in range(len(chars))]
			write_file.write(json.dumps(char_inds) + '\n')
			line = read_file.readline()
			if line_num % 5000 == 0:
				print('Processing line ' + str(line_num))
			line_num += 1
		read_file.close()
	write_file.close()

