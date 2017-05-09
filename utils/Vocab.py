import configparser
import json
import operator
import os
import pickle

from . import string_utils

base = os.path.dirname(__file__)

config = configparser.ConfigParser()
config.read(base + '/../config/conf.cf')


class Vocab():
    def __init__(self, filepath=None, item_freqs=None, parse_filepath=None, char=False, lower=True,
                 include_special=False, cutoff=None, top_n=None, accessor=None):
        """
        An object representing a vocabulary object. Should not be altered once finalized.
        Lifespan: Instantiation -> Finalized (if given item_freqs or parse_filepath)
        Instantiation -> Process -> Call Finalize
        
        :param filepath: Filepath to save the object in
        :param item_freqs: Item frequencies to initialize the vocabulary to
        :param parse_filepath: Filepath to parse from
        :param char: Whether or not to use characters (True) or words (False)
        :param lower: Whether or not to lowercase vocabulary
        :param include_special: Whether or not to include special characters
        :param cutoff: Minimum frequency of word before cutting it off
        """

        self.filepath = filepath
        self.char = char
        self.lower = lower
        self.cutoff = cutoff or 0
        self.top_n = top_n or 1000000000
        self.include_special = include_special
        self.finalized = False
        self.size = 0
        self.accessor = accessor

        if item_freqs:
            self.vocab = {}
            self.interp = {}
            self.freqs = item_freqs
            self.finalize()
        elif parse_filepath:
            clean_dataset_dir = base + '/.' + config['FILE LOCS']['clean_dataset_dir']
            filepaths = [clean_dataset_dir + '/' + filep for filep in os.listdir(clean_dataset_dir) if
                         filep.endswith('.txt')]
            for filepath in filepaths:
                self._parse_json_file_vocab(filepath)
            self.finalize()
        else:
            self.vocab = {}
            self.freqs = {}
            self.interp = {}

        self.new_vocab = {}
        self.new_interp = {}

    def _parse_json_file_vocab(self, filepath, accessor=None):
        """
        Parse vocab from the file specified in the filepath. Assumes lines are JSONs, and retrieves strings to parse
        using the function accessor.

        :param filepath: Filepath to parse from
        :param accessor: Function that retrieves string to parse from the JSON
        :return: 
        """
        accessor = accessor or self.accessor
        read_file = open(filepath, 'r')
        json_str = read_file.readline()
        print('Parsing vocab from ' + filepath)
        while json_str:
            self._extract_from_json(json_str, accessor)

    def _extract_from_json(self, json_string, accessor=None):
        """
        Parses and adds vocab from the JSON string, and retrieves string to parse using the function accessor.
        
        :param json_obj: JSON string to extract string from
        :param accessor: Function that retrieves string to parse from the JSON
        :return: None
        """
        accessor = accessor or self.accessor
        json_obj = json.loads(json_string)
        self._extract_from_json_obj(json_obj, accessor)

    def _extract_from_json_obj(self, json_obj, accessor=None):
        """
        Parses and adds vocab from the JSON object, and retrieves string to parse using the function accessor.
        
        :param json_obj: JSON object to extract string from
        :param accessor: Function that retrieves string to parse from the JSON
        :return: None
        """
        accessor = accessor or self.accessor
        to_extract = accessor(json_obj)
        if not to_extract:
            return
        else:
            self._extract_from_string(to_extract)

    def _extract_from_string(self, to_extract):
        """
        Parses out the string into items
        
        :param to_extract: String to extract vocab from
        :return: 
        """
        if not to_extract:
            return
        to_parse = to_extract.encode().decode()

        if self.char:
            items = string_utils.parse_string_to_chars(to_parse, self.lower, self.include_special)
        else:
            items = string_utils.parse_string_to_words(to_parse, self.lower, self.include_special)

        for item in items:
            self._item_seen(item)

    def _item_seen(self, item):
        """
        Marks an item as having been encountered.
        
        :param item: Item that has been seen.
        :return: None
        """
        if item in self.freqs.keys():
            self.freqs[item] += 1
        else:
            self.freqs[item] = 1

    def decode(self, tokens):
        """
        Decodes list of tokens into list of vocab items.
        
        :param tokens: List of tokens
        :return: List of vocab items
        """
        return [self.interp[token] for token in tokens]

    def encode(self, items):
        """
        Encodes list of vocab items into list of tokens
        
        :param items: List of vocab items or string to encode
        :return: List of tokens
        """
        if type(items) is type([]):
            return [self.vocab[item] for item in items]
        elif type(items) is type('string'):
            if self.char:
                items = string_utils.parse_string_to_chars(items, self.lower, self.include_special)
            else:
                items = string_utils.parse_string_to_words(items, self.lower, self.include_special)
            return [self.vocab[item] for item in items]

    def finalize(self):
        """
        Marks this vocabulary as not editable. Extracts vocab and interp from the vocab frequencies.
        
        :return: Vocab (self) 
        """
        sorted_vocab_list = sorted(self.freqs.items(), key=operator.itemgetter(1), reverse=True)

        # Note that this is only okay because the frequencies have already been sorted from largest to smallest.
        # After the first item that is below the cutoff, no more items are added to the keys anymore.
        active_vocab = self.new_vocab if self.finalized else self.vocab
        active_interp = self.new_interp if self.finalized else self.interp
        for i in range(min(len(sorted_vocab_list, self.top_n))):
            if sorted_vocab_list[i][1] > self.cutoff:
                active_vocab[sorted_vocab_list[i][0]] = i + 5

        active_vocab['<Z>'] = 0
        active_vocab['<UNK>'] = 1
        active_vocab['<S>'] = 2
        active_vocab['</S>'] = 3
        active_vocab['<||>'] = 4

        for item in active_vocab:
            active_interp[active_vocab[item]] = item

        self.size = len(self.vocab)

        self.finalized = True
        return self

    def force_safe_update(self):
        """
        Forces a safe update that doesn't override the current bindings
        
        :return: Vocab (self) 
        """
        curr_ind = self.size
        for item in self.new_vocab.keys():
            if item not in self.vocab.keys():
                self.vocab[item] = curr_ind
                curr_ind += 1
        for item in self.vocab:
            self.interp[self.vocab[item]] = item
        self.size = len(self.vocab)
        return self

    def force_unsafe_update(self):
        """
        Forces an unsafe update that can override the current bindings
        
        :return: Vocab (self) 
        """
        self.finalized = False
        self.finalize()
        return self

    def get_updated_vocab(self, new_filepath):
        """
        Returns new vocab object. Use this for a new updated vocab object based on updated frequencies. Note that
        some formerly present vocabulary may no longer be present due to updates. This is unsafe; bindings may be changed.
        
        :param new_filepath: New save filepath
        :return: New Vocab object based on updated frequencies
        """
        return Vocab(filepath=new_filepath, item_freqs=self.freqs, char=self.char, lower=self.lower,
                     include_special=self.include_special, cutoff=self.cutoff, top_n=self.top_n)

    def size(self):
        return self.size

    def vocab_set(self):
        return self.vocab.keys()

    def encoder(self):
        return dict(self.vocab)

    def decoder(self):
        return dict(self.interp)

    def save(self, filepath=None):
        """
        Takes in the absolute filepath and saves the vocab to that path with the name.
        
        :param filepath: Absolute filepath to save to
        :return: Nothing
        """
        if not filepath:
            filepath = self.filepath
        assert filepath is not None, 'Filepath not specified'
        pickle.dump(self, open(filepath, 'wb+'))
