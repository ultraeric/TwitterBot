import configparser
import os

from utils import parse_utils as puts

base = os.path.dirname(__file__)

config = configparser.ConfigParser()
config.read(base + '/../config/conf.cf')


def parse_vocab():
    clean_dataset_dir = base + '/.' + config['FILE LOCS']['clean_dataset_dir']
    read_filepaths = [clean_dataset_dir + '/' + filep for filep in os.listdir(clean_dataset_dir) if
                      filep.endswith('.txt')]
    write_filepaths = [base + '/.' + config['FILE LOCS']['vocab_dir'] + '/vocab.data']
    lower = [False]
    char = [True]
    include_special = [True]
    json_to_string_accessors = [lambda json_obj: json_obj['text']]
    min_cutoffs = [int(float(config['DATA INFO']['min_cutoffs']))]
    puts.parse_vocab(read_filepaths, write_filepaths, json_to_string_accessors, lower=lower, char=char,
                     include_special=include_special, min_cutoffs=min_cutoffs)
