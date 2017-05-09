import configparser
import os

import torch_model

base = os.path.dirname(os.path.abspath(__file__))
start_batch = 0

config = configparser.ConfigParser()
config.read(base + '/config/conf.cf')

model = torch_model.Seq2Seq()
model.cuda()
