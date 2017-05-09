import os
import re

base = os.path.dirname(__file__)


def pad_token_arrays(token_arrays, max_len, pad_token, pad_front=True):
    """
    Pads the 2-D array token_arrays with pad_token to be a full 2-D array with equal length subarrays. 
    By default, pads the beginning of the array. To pad the end, use pad_front=False.
    Cuts off the strings at the max_len. The original array is not affected.
    
    :param token_arrays: 2-D array to pad
    :param max_len: Max size of sub-array
    :param pad_token: Token to pad 2-D array with
    :param pad_front: Whether or not to pad the front (True) or back (False)
    :return: Returns the equal dimension 2-D array
    """

    # Calculates the max length of sub-arrays
    max_len = min(max([len(token_array) for token_array in token_arrays]), max_len)

    for i in range(len(token_arrays)):

        # If the sub-array is not at the max_length, pad it. Creates new sub-lists and does not alter originals.
        if len(token_arrays[i]) < max_len:
            pad = [pad_token] * (max_len - len(token_arrays[i]))
            if pad_front:
                token_arrays[i] = pad + token_arrays[i]
            else:
                token_arrays[i] = token_arrays[i] + pad
        else:
            token_arrays[i] = token_arrays[i][:max_len]

    return token_arrays


def parse_string_to_chars(string, lower=True, include_special=True):
    """
    Parses a string to a list of characters, from raw json encoded to UTF-8 encoding.
    
    :param string: Raw string with unparsed UTF-8
    :param lower: Whether or not to lowercase items
    :param include_special: Whether or not to include special characters
    :return: List of parsed characters
    """

    #Goes from raw encoding to UTF-8 encoding
    string = string.encode().decode()
    if lower:
        string = string.lower()
    if not include_special:
        string = re.sub(r'[\W]+', '')
    return [string[i] for i in range(len(string))]


def _parse_string_to_words(string, lower=True, include_special=True):
    """
    Parses a string to a list of words, from raw json encoded to UTF-8 encoding.
    
    :param string: Raw string with unparsed UTF-8
    :param lower: Whether or not to lowercase items
    :param include_special: Whether or not to include special characters
    :return: List of parsed words
    """
    string = string.encode().decode()
    if lower:
        string = string.lower()
    if include_special:
        separate_special = re.sub(r'(\W)+', r' \1 ', string)
        return re.split(r'[\s]+', separate_special)
    else:
        return re.split(r'[\W]+', string)


def parse_string_to_words(string, lower=True, include_special=True):
    """
    Helper function for _parse_string_to_words. Not sure why this currently exists 🤔🤔🤔.
    """
    return _parse_string_to_words(string, lower, include_special)


def parse_string_to_grammar_tokens(string, lower=True, include_special=True):
    return _parse_string_to_words(string, lower, include_special)
