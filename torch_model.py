import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as functional
import pickle
import os
import configparser

base = os.path.dirname(os.path.abspath(__file__))

config = configparser.ConfigParser()
config.read(base + '/config/conf.cf')

def model_param(key):
    return int(float(config['MODEL INFO'][key]))

class Seq2Seq(nn.Module):

    def __init__(self):
        super().__init__()

        self.vocab = pickle.load(open(base + '/' + config['FILE LOCS']['vocab_dir'] + '/vocab.data', 'rb'))
        self.vocab_size = len(self.vocab[0]) + 5

        self.state_size, self.embed_size = model_param('state_size'), model_param('embedding_size'),

        # LOOKUP TABLE
        # Lookup tables for the embeddings of characters
        # Note that the L2-norm is automatically applied
        # Input: [batch_size, seq_length (ids of words)]
        # Output: [batch_size, seq_length, embed_size]
        self.vocab_lookup = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        # LSTM CELLS
        # Faster than LSTM operation, but a bit more complex to set up
        # Input: [batch_size, seq_length, embed_size]
        # Output: [batch_size, seq_length, state_size]
        self.init_lstm = nn.LSTMCell(input_size=self.embed_size, hidden_size=self.state_size)
        self.hidden_cells = [nn.LSTMCell(input_size=self.state_size, hidden_size=self.state_size)
                             for _ in range(model_param('gru_depth') - 1)]

        # LINEAR
        # Input: [batch_size * seq_length, state_size]
        # Output: [batch_size * seq_length, vocab_size]
        self.output_space_transform = nn.Linear(in_features=self.state_size, out_features=self.vocab_size)

    def forward(self, input):
        input = self.vocab_lookup(input)
        input = self.init_lstm(input)
        for i in range(self.state_size):
            input = self.hidden_cells[i](input)
        input = self.output_space_transform(input.view(-1, self.state_size))
        input = functional.log_softmax(input).view(self.init_lstm.size()[0], self.init_lstm.size()[1], self.vocab_size)
        return input