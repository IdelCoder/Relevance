#_*_coding:utf-8_*_
# This file is self-contained.
# The training data are randomly generated.
#


import numpy as np
import tensorflow as tf
import random
import math

LEARNING_RATE = 0.001
MAX_CHARS = 15
TOL_PARAM, TOL_LOSS, TOL_GRAD = 1e-6, 1e-6, 1e-6
BATCH_SIZE = 100
EPOCH = 10000

SAMPLE_SIZE_POS = 500
SAMPLE_SIZE_NEG = 1000
SAMPLE_SIZE = SAMPLE_SIZE_POS + SAMPLE_SIZE_NEG

SERVICE_SIZE = 500
MASHUP_SIZE = 100

DIM = 50
W1_FILE = "w1.txt"

# This is the parameter to learn
W1n = np.random.normal(size=(DIM, DIM))


# generate sevice sets and mashup sets
service_lst = np.random.normal(size=(DIM, SERVICE_SIZE))

mashup_lst = np.random.normal(size=(DIM, MASHUP_SIZE))


# Sorry for my pool coding
# The following codes construct training data
# and converting them into numpy format for the convenience of training

# generate positive examples
temp_pos_n = 0

SAMPLE_POS_S = np.array(service_lst[:, 0])
SAMPLE_POS_M = np.array(mashup_lst[:, 0])

SAMPLE_POS_RECORD = {}
while temp_pos_n < SAMPLE_SIZE_POS - 1:
	service_i = random.randint(0, SERVICE_SIZE - 1)
	mashup_i = random.randint(0, MASHUP_SIZE - 1)
	if SAMPLE_POS_RECORD.has_key(service_i):
		if mashup_i in SAMPLE_POS_RECORD[service_i]:
			continue
		SAMPLE_POS_RECORD[service_i].append(mashup_i)
	else:
		SAMPLE_POS_RECORD[service_i] = [mashup_i]
	if temp_pos_n == 0:
		SAMPLE_POS_S = np.append([SAMPLE_POS_S], [np.array(service_lst[:, service_i])], axis=0)
		SAMPLE_POS_M = np.append([SAMPLE_POS_M], [np.array(mashup_lst[:, mashup_i])], axis=0)
	else:
		SAMPLE_POS_S = np.append(SAMPLE_POS_S, [np.array(service_lst[:, service_i])], axis=0)
		SAMPLE_POS_M = np.append(SAMPLE_POS_M, [np.array(mashup_lst[:, mashup_i])], axis=0)
	
	temp_pos_n += 1
# generate negative examples
SAMPLE_NEG_S = np.array(service_lst[:, 1])
SAMPLE_NEG_M = np.array(mashup_lst[:, 1])
temp_neg_n = 0
while temp_neg_n < SAMPLE_SIZE_NEG - 1:
	service_i = random.randint(0, SERVICE_SIZE - 1)
	mashup_i = random.randint(0, MASHUP_SIZE - 1)
	if SAMPLE_POS_RECORD.has_key(service_i) and mashup_i in SAMPLE_POS_RECORD[service_i]:
		continue

	if temp_neg_n == 0:
		SAMPLE_NEG_S = np.append([SAMPLE_NEG_S], [np.array(service_lst[:, service_i])], axis=0)
		SAMPLE_NEG_M = np.append([SAMPLE_NEG_M], [np.array(mashup_lst[:, mashup_i])], axis=0)
	else:
		SAMPLE_NEG_S = np.append(SAMPLE_NEG_S, [np.array(service_lst[:, service_i])], axis=0)
		SAMPLE_NEG_M = np.append(SAMPLE_NEG_M, [np.array(mashup_lst[:, mashup_i])], axis=0)
	temp_neg_n += 1

# 
# Merge training and testing set of services and mashups
SAMPLE_S = np.row_stack((SAMPLE_POS_S, SAMPLE_NEG_S))
SAMPLE_M = np.row_stack((SAMPLE_POS_M, SAMPLE_NEG_M))

# assign 0 to positive examples and -1 to negative examples as labels
# This is related to the way that loss function works
labels = np.array([0 if n < SAMPLE_SIZE_POS else -1 for n in range(SAMPLE_SIZE)])


# s is short for service，m for mashup. y are labels
s = tf.placeholder(dtype=tf.float32)
m = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)


W1 = tf.Variable(initial_value=W1n, dtype=tf.float32)

# P(y=1;W1)
def prob_pos(weight, s, m):
	result = 0.0
	with tf.Session() as sess:
		# Ws = tf.matmul(tf.expand_dims(tu[0],0), weight)
		Ws = tf.reduce_sum(tf.multiply(tf.expand_dims(s,-1), weight), axis=0)
		sigWs = tf.sigmoid(Ws)
		result = tf.tensordot(sigWs, m, 1)
	return tf.sigmoid(result)

# P(y=0;W1)
def prob_neg(weight, s, m):
	return 1.0 - prob_pos(weight, s, m)

# The loss function (for mini-batch)
# If you are fresh to tensorflow, you might think of the loss function strange.
# Since you cannot use 'for' or other iteration operators on the Tensor Object, you have to explore
# a work round to compute the losses of the batch in the view of matrix computing.
# Make a trial and you will find where the results are located.
def f_myloss(y=y, s=s, m=m, weight=W1):
	sigWs = tf.sigmoid(tf.matmul(a=weight, b=s, transpose_b=True))
	result = tf.sigmoid(tf.matmul(a=m, b=sigWs))
	# get the diagonal
	diag = tf.diag_part(result)
	# plus y (i.e., lables)
	add_y = tf.abs(tf.add(diag, y))
	return add_y



myloss = f_myloss(y=y, s=s, m=m, weight=W1)

neg_log_likelihood = -1.0 * (tf.reduce_sum(myloss) - tf.nn.l2_loss(W1))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

train_op = optimizer.minimize(loss=neg_log_likelihood)

grad = tf.gradients(neg_log_likelihood, W1)

with tf.Session() as sess:

	sess.run(fetches=tf.global_variables_initializer())
	# number of batchs in one epoch
	n_batch = int(math.ceil(float(SAMPLE_SIZE) / BATCH_SIZE))
	
	obs_W1 = []
	obs_loss = []
	obs_grad = []
	j = 1
	while True:
		new_loss = 0.0
		new_grad = 0.0
		new_W1 = 0.0
		for i in range(n_batch):
			
			# 50*BATCH_SIZE
			s_batch = SAMPLE_S[BATCH_SIZE * i: BATCH_SIZE * (i+1)]
			# 50*BATCH_SIZE
			m_batch = SAMPLE_M[BATCH_SIZE * i: BATCH_SIZE * (i+1)]
			# 100
			y_batch = labels[BATCH_SIZE * i: BATCH_SIZE * (i+1)]
			
			feed_dict = {s: s_batch, m: m_batch, y: y_batch}
			sess.run(fetches=train_op, feed_dict=feed_dict)
			new_W1 = sess.run(fetches=W1)
			new_loss = sess.run(fetches=neg_log_likelihood, feed_dict=feed_dict)
			new_grad = sess.run(fetches=grad, feed_dict=feed_dict)
		obs_W1.append(new_W1)
		obs_loss.append(new_loss)
		obs_grad.append(new_grad)
		if len(obs_loss) < 2:
			continue
		new_loss = obs_loss[-1]
		old_loss = obs_loss[-2]
		loss_diff = np.abs(new_loss - old_loss)
		if loss_diff < TOL_LOSS:
			np.savetxt(W1_FILE, new_W1)
	 		print('Loss function convergence in {} iterations!'.format(j))
	 		break
	 	else:
	 		print('iterations: %d, loss_diff: %f') % (j, loss_diff)
	 	if j >= EPOCH:
	 		np.savetxt(W1_FILE, new_W1.eval())
	 		print('Loss function cannot convergence!')
	 		break
	 	j += 1
	
