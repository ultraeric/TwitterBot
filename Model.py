import tensorflow as tf
import tflearn as tfl
import configparser
import pickle

base = os.path.dirname(__file__)


def model_param(key):
	return int(float(config['MODEL INFO'][key]))

def _create_variable(name, shape = None, initializer = tf.contrib.layers.xavier_initializer(), regularizer = None):
	"""Helper function. Creates/gets a variable using some optional info.
	Args:
	  name <string>: name of Variable
	  shape <list of ints>: shape of Variable
	  initializer <TF initializer>
	  regularizer <TF regularizer>: tf.contrib.layers.regularizers
	"""
	return tf.get_variable(name, shape = shape, dtype = tf.float32, initializer = initializer, regularizer = regularizer)

def build_model(gpu = True, gpu_num = 0):
	"""Outward-facing method builds and returns method when called on either GPU or CPU depending
	on the (gpu) parameter.
	Args:
	  gpu <boolean> whether or not use gpu for the model
	"""

	model = None
	if gpu:
		with tf.device('/gpu:' + str(gpu_num)):
			model = _build_model()
	else:
		model = _build_model()
	return model

def _build_model():
	"""Convenience method builds and returns the model when called"""

	config = configparser.ConfigParser()
	config.read(base + '/../config/conf.cf')
	vocab = pickle.load(open(base + '/.' + config['FILE LOCS']['vocab_dir'] + '/vocab.data', 'rb'))	
	vocab_size = len(vocab[0]) + 5

	#PARAMETERS
	#Lookup tables for the embeddings of characters
	#Sizes: [vocab_size x embed_size]
	vocab_lookup = _create_variable(name = 'vocab_lookup', 
					  shape = [vocab_size, 
						   model_param('embedding_size')], 
					  regularizer = tf.contrib.layers.l2_regularizer(1.0))
	
	#TENSORS
	#Placeholders for input of the model. Labels represent target.
	#Sizes: [batch_size x timesteps]
	sequence_ids = tf.placeholder(dtype = tf.int32, shape = [None, None])
	labels = tf.placeholder(dtype = tf.int32, shape = [None, None])	

	#Flatten labels for later softmaxing.
	labels_reshaped = tf.reshape(labels, [-1])

	#OPERATIONS
	#Get input tensors from the lookup tables.
	#Input: [batch_size x time_steps]
	#Output: [batch_size x time_steps x embed_size]
	sequence_embeds = tf.nn.embedding_lookup(vocab_lookup, sequence_ids)
	
	#CELL
	#Build the GRU object. This creates the parameters, computation
	#graph will be created it later.
	gru_cell = tf.nn.rnn_cell.GRUCell(model_param('state_size')))
	gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, output_keep_prob = float(config['MODEL INFO']['dropout_keep_prob']))
	gru_cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * model_param('gru_depth'))

	#OPERATION
	#An operation wraps the GRU cell to turn it into an operation for the computational graph
	#Input: [batch_size x time_steps x embed_size]
	#Output: [batch_size x time_steps x gru_state_size] and [batch_size x gru_state_size]
	gru_output, gru_final_state = tf.nn.dynamic_rnn(gru_cell, sequence_embeds, dtype = tf.float32)

	#Flatten GRU output for batch learning + softmaxing
	gru_output = tf.reshape(gru_output, [-1, model_param('state_size')])

	#OPERATION
	#An operation that brings the GRU output back into output space.
	#Input: [n x gru_state_size]
	#Output: [n x vocab_size]
	sequence_output = tfl.fully_connected(gru_output, vocab_size, regularizer = 'L2')	

	#Brings last output of GRU to i/o space and dimensions to allow for evaluation.
	eval_output = tf.reshape(tf.nn.log_softmax(tf.unstack(sequence_output, axis = 1)[-1]), [int(float(config['TRAIN INFO']['batch_size'])), -1, vocab_size])
	#Gets top k probabilities. 
	top_k_probs, top_k_inds = tf.nn.top_k(eval_output, k = 10)

	#OPERATION
	#Count the number of non-zeros that the input has to normalize by number of words.
	num_nonzero = tf.cast(tf.count_nonzero(sequence_ids), tf.float32)
	
	#OPERATION
	#Generate the zero-masking for the output.
	zero = tf.constant(0, dtype=tf.float32)
	where = tf.cast(sequence_ids, dtype = tf.float32)
	zero_mask = tf.cast(tf.cast(tf.reshape(tf.not_equal(where, zero), [-1]), dtype = tf.int32), dtype = tf.float32)
	
	#OPERATION 
	#Generate the loss function.
	loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(sequence_output, labels_reshaped) * zero_mask) / num_nonzero + float(config['TRAIN INFO']['lambda']) * sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
	train_step = tf.train.AdamOptimizer().minimize(loss)
	return {'train_step': train_step, 
		'loss': loss, 
		'vocab_lookup': vocab_lookup, 
		'sequence_ids': sequence_ids, 
		'labels': labels, 
		'sequence_embeds': sequence_embeds, 
		'gru_cell': gru_cell, 
		'gru_output': gru_output, 
		'gru_final_state': gru_final_state, 
		'sequence_output': sequence_output, 
		'num_nonzero': num_nonzero, 
		'zero_mask': zero_mask,
		'eval_output': eval_output,
		'top_k_probs': top_k_probs,
		'top_k_inds': top_k_inds}
		

