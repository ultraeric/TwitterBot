import datasets
import vocab

vocab.parse_vocab()

datasets.clean()
datasets.process()