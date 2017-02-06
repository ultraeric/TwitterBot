import vocab.Vocab_Parser
import datasets.Cleaner
import datasets.Preprocessor

vocab.Vocab_Parser.parse_vocab()
datasets.Cleaner.clean()
datasets.Preprocessor.process()

