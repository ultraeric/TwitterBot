import os
import json
import gzip
import re
import configparser
from utils import string_utils as suts

base = os.path.dirname(__file__)

config = configparser.ConfigParser()
config.read(base + '/../config/conf.cf')

def clean():
	dirty_dir = base + '/.' + config['FILE LOCS']['dirty_dataset_dir']
	file_num = 0 
	print(dirty_dir)
	for filename in os.listdir(dirty_dir):
		if not filename.endswith('.gz'):
			continue
		zip_file = gzip.open(dirty_dir + '/' + filename, 'rt')
		line = zip_file.readline()

		new_file = open(base + '/.' + config['FILE LOCS']['clean_dataset_dir'] + '/cleaned' + str(file_num) + '.txt', 'w+')
		print(filename)

		while line:
			if not line:
				continue
			json_obj = None
			try:
				json_obj = json.loads(line)
			except:
				line = zip_file.readline()
				continue
			if 'text' not in json_obj:
				line = zip_file.readline()
				continue
			text = json_obj['text'].encode().decode().replace('\n', '\\n')
			text = re.sub(r'http\S+', '', text)
			text = re.sub(r'&amp;', '&', text)
			dic = {"text": text}
			new_file.write(json.dumps(dic) + '\n')
			line = zip_file.readline()
		file_num += 1
		new_file.close()			
