def calc_recall(tokens, target_tokens):
    if len(tokens) == 0:
        return 0
    return sum([1 if token in target_tokens else 0 for token in tokens]) / len(tokens)


def calc_precision(tokens, target_tokens):
    if len(target_tokens) == 0:
        return 1
    return sum([1 if token in target_tokens else 0 for token in tokens]) / len(target_tokens)


def calc_f1(tokens, target_tokens):
    precision = calc_precision(tokens, target_tokens)
    recall = calc_recall(tokens, target_tokens)
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)
