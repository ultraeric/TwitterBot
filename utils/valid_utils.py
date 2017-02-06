

def calc_recall(tokens, target_tokens):
	return sum([1 if token in target_tokens else 0 for token in tokens]) / len(tokens)

def calc_precision(tokens, target_tokens):
	return sum([1 if token in target_tokens else 0 for token in tokens]) / len(target_tokens)

def calc_f1(tokens, target_tokens):
	precision = calc_precision(tokens, target_tokens)
	recall = calc_recall(tokens, target_tokens)
	return 2 * precision * recall / (precision + recall)	
