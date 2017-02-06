import tensorflow as tf
import tflearn as tfl
import Model
import configparser
import os
import datasets.Batch_Maker

base = os.path.dirname(os.path.abspath(__file__))
start_batch = 0

#Parse configuration files
config = configparser.ConfigParser()
print(base)
config.read(base + '/config/conf.cf')

load_model = base + '/' + config['FILE LOCS']['model_load']
load_model = None

#Builds and retrieves model on a GPU
model = Model.build_model(gpu = True, gpu_num = 1)
train_step = model['train_step']

for key in model:
	tf.add_to_collection(key, model[key])

#Dynamic on-GPU and off-GPU computing (for GPU-incompatible things)
saver = tf.train.Saver()
configp = tf.ConfigProto(allow_soft_placement = True)
configp.gpu_options.allocator_type = 'BFC'
configp.gpu_options.allow_growth = True

tf.logging.set_verbosity(tf.logging.FATAL)

#Train loop
with tf.Session(config = configp) as sess:
	#Initializes global variables
	sess.run(tf.global_variables_initializer())

	if load_model:
		keys = [key for key in model]
		model = {}
		new_saver = tf.train.import_meta_graph(load_model + '.meta', clear_devices = True)

		with tf.device('/gpu:1'):
			new_saver.restore(sess, tf.train.latest_checkpoint('./'))
		for key in keys:
			item = tf.get_collection(key)
			print(item)
			if item:
				print(key)			
				model[key] = item[0]

	#Gets the batch generator function
	batch_func = datasets.Batch_Maker.get_batch_func(gpu = True, gpu_num = 1)
	batch = batch_func()
	batch_num = 0
	
	#While a batch exists in the current file, train
	while batch:
		print('Working on batch ' + str(batch_num))
		if batch_num < start_batch:
			batch_func()
			batch_num += 1
			continue
		train_step.run(feed_dict = {model['sequence_ids']: batch[0],
					model['labels']: batch[1]})	
		batch = batch_func()
		batch_num += 1
		if batch_num % 5000 == 0:
			saver.save(sess, base + '/' + config['FILE LOCS']['model_save'])
	#Save model
	saver.save(sess, base + '/' + config['FILE LOCS']['model_save'])
