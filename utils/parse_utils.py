import json
import os

base = os.path.dirname(__file__)
import configparser
from .vocab import Vocab

config = configparser.ConfigParser()
config.read(base + '/../config/conf.cf')


def parse_vocab(read_filepaths, write_filepaths, json_to_string_accessors, char=None, lower=None,
                include_special=None, min_cutoffs=None, top_n=None):
    """
    Parses all the items in the filepaths to vocabs and interpretations. Expects each line in the files in the filepath
    to be in JSON format. Extracts the strings to parse from the json according to the json_to_string_accessors.
    By default parses by character.
    
    :param filepaths: Filepaths to search for files to parse vocab from
    :param write_filepaths: Filepaths to write vocab to
    :param json_to_string_accessors: Accessors for the strings to parse from the JSONs
    :param char: Whether to parse by character. Default is True
    :param lower: Whether to lowercase the vocab. Default is True
    :param include_special: Whether to include special characters. Default is True
    :param min_cutoffs: The minimum cutoff for the vocabulary to parse. Default is none.
    :return: None
    """
    assert len(write_filepaths) == len(
        json_to_string_accessors), 'Number of vocabulary items is not the same as number of json accessors'

    char = char or [False for _ in range(len(write_filepaths))]
    lower = lower or [True for _ in range(len(write_filepaths))]
    include_special = include_special or [False for _ in range(len(write_filepaths))]
    min_cutoffs = min_cutoffs or [0 for _ in range(len(write_filepaths))]
    top_n = top_n or 1000000000

    vocabs = [Vocab(filepath=write_filepaths[i],
                    char=char[i],
                    lower=lower[i],
                    include_special=include_special[i],
                    cutoff=min_cutoffs[i],
                    accessor=json_to_string_accessors[i],
                    top_n=top_n)
              for i in range(len(write_filepaths))]

    for read_filepath in read_filepaths:
        read_file = open(read_filepath, 'r')
        json_str = read_file.readline()
        print(read_file)
        while json_str:
            json_obj = json.loads(json_str)
            for vocab in vocabs:
                vocab._extract_from_json_obj(json_obj)
            json_str = read_file.readline()
    for vocab in vocabs:
        vocab.finalize()
        vocab.save()
