import tensorflow as tf

import model as Model

load_model = './models/model'

model = Model.build_model(gpu=True)

saver = tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
tf.logging.set_verbosity(tf.logging.FATAL)

with tf.Session(config=config) as sess:
    keys = [key for key in model]
    model = {}
    new_saver = tf.train.import_meta_graph(load_model + '.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    for key in keys:
        item = tf.get_collection(key)
        if item:
            print(key)
            model[key] = item[0]
