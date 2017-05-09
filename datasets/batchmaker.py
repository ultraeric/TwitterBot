import configparser
import json
import os

from utils import string_utils as suts

base = os.path.dirname(__file__)

config = configparser.ConfigParser()
config.read(base + '/../config/conf.cf')


def get_token(key):
    return int(float(config['VOCAB INFO'][key]))


zero_mask = get_token('zero_mask')
unk_token = get_token('unk_token')
start_token = get_token('start_token')
end_token = get_token('end_token')
div_token = get_token('div_token')


def _get_batch_func(use_tf, processed_filename=config['FILE LOCS']['preprocessed_dataset'], gpu=False, gpu_num=0):
    data_file = open(base + '/.' + processed_filename, 'r')

    def _next_batch():
        batch_in = []
        batch_out = []
        for i in range(int(float(config['TRAIN INFO']['batch_size']))):
            line = data_file.readline()
            if not line:
                return None
            vocab_inds = json.loads(line)
            batch_in.append([start_token] + vocab_inds)
            batch_out.append(vocab_inds + [end_token])
        return (
            suts.pad_token_arrays(batch_in, 200, zero_mask, False),
            suts.pad_token_arrays(batch_out, 200, zero_mask, False))

    def next_batch():
        if use_tf:
            import tensorflow as tf
            if gpu:
                with tf.device('/gpu:' + str(gpu_num)):
                    return _next_batch()
            else:
                return _next_batch()
        else:
            return _next_batch()

    return next_batch


def get_tf_batch_func(processed_filename=config['FILE LOCS']['preprocessed_dataset'], gpu=False, gpu_num=0):
    return _get_batch_func(True, processed_filename, gpu, gpu_num)


def get_torch_batch_func(processed_filename=config['FILE LOCS']['preprocessed_dataset']):
    return _get_batch_func(False, processed_filename)
