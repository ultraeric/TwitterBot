import os
base = os.path.dirname(__file__)

def pad_token_arrays(token_arrays, max_len, pad_token, pad_front = True):
	max_len = min(max([len(token_array) for token_array in token_arrays]), max_len)
	for i in range(len(token_arrays)):
		if len(token_arrays[i]) < max_len:
			pad = [pad_token] * (max_len - len(token_arrays[i]))
			if pad_front:
				token_arrays[i] = pad + token_arrays[i]
			else:
				token_arrays[i] = token_arrays[i] + pad
		else:
			token_arrays[i] = token_arrays[i][:max_len]
	return token_arrays

def parse_string_to_chars(string, lower = True, include_special = True):
	string = string.encode().decode()
	if lower:
		string = string.lower()
	if not include_special:
		string = re.sub(r'[\W]+', '')
	return [string[i] for i in range(len(string))]

def _parse_string_to_words(string, lower = True, include_special = True):
	string = string.encode().decode()
	if lower:
		string = string.lower()
	if include_special:
		separate_special = re.sub(r'(\W)+', r' \1 ', string)
		return re.split(r'[s]+', separate_special)
	else:
		return re.split(r'[\W]+', string)

def parse_string_to_words(string, lower = True, include_special = True):
	return _parse_string_to_words(string, lower, include_special)

def parse_string_to_grammar_tokens(string, lower = True, include_special = True):
	return _parse_string_to_words(string, lower, include_special)
