import configparser
import json
import os
import pickle

import utils.Vocab.Vocab as Vocab

base = os.path.dirname(__file__)

config = configparser.ConfigParser()
config.read(base + '/../config/conf.cf')

# Load vocab from file
vocab = pickle.load(open(base + '/.' + config['FILE LOCS']['vocab_dir'] + '/vocab.data', 'rb'))
assert type(vocab) is Vocab, 'Loaded vocab is not Vocab object'


def preprocess():
    """
    Takes cleaned dataset and preprocesses it into a tokenized, fully preprocessed dataset.
    
    :return: None
    """
    write_file = open(config['FILE LOCS']['preprocessed_dataset'], 'w+')

    clean_dataset_dir = base + '/.' + config['FILE LOCS']['clean_dataset_dir']
    for filename in os.listdir(clean_dataset_dir):
        if not filename.endswith('.txt'):
            continue
        print(filename)
        read_file = open(clean_dataset_dir + '/' + filename, 'r')
        json_str = read_file.readline()
        while json_str:
            to_parse = json.loads(json_str)['text'].encode().decode().replace('\\n', '')
            char_inds = vocab.encode(to_parse)
            write_file.write(json.dumps(char_inds) + '\n')
            json_str = read_file.readline()
        read_file.close()
    write_file.close()
