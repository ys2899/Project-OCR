import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import tensorflow as tf

from datetime import datetime
from scipy.io import loadmat
from sklearn.utils import shuffle



def get_transformed_data():
	print "Reading in and transforming data..."
	df = pd.read_csv('../large_files/train.csv')
	data = df.as_matrix().astype(np.float32)
	np.random.shuffle(data)
	X = data[:, 1:]
	mu = X.mean(axis=0)
	X = (X - mu)/255; # center the data and normalize them.
	# Getting the dimension of the data
	N = X.shape[0];
	L = np.sqrt(len(X[1,])).astype(int)
	X = np.reshape(X, [ N, L, L, 1])
	Y = data[:, 0]

	return X, Y, N


def y2indicator(y):
	N = len(y)
	ind = np.zeros((N,10))
	for i in xrange(N):
		ind[i,y[i]] = 1
	return ind


def error_rate(p,t):
	return np.mean(p != t);


def convpool(X,W,b):
	conv_out = tf.nn.conv2d(X,W,strides = [1,1,1,1], padding= 'SAME');
	conv_out = tf.nn.bias_add(conv_out, b);
	pool_out = tf.nn.max_pool(conv_out, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	return pool_out;


# # # Shape will be defined appropriately.


def init_filter(shape, poolsz):
	w = np.random.randn(*shape)/ np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
	return w.astype(np.float32);

def main():

	X, Y, N = get_transformed_data();
	print "Performing logistic regression..."

	Xtrain = X[:58000,]
	Ytrain = Y[:58000]
	
	Xtest  = X[-1000:,]
	Ytest  = Y[-1000:]

	Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
	Ytrain_ind = y2indicator(Ytrain.astype(int))
	Ytest_ind  = y2indicator(Ytest.astype(int))

	max_iter = 5;
	print_period = 10;
	batch_sz = 500;
	K = 10;
	n_batches = 58000 / batch_sz;


	# initialize the weights:
	M = 500;
	poolsz = (2,2);


	W1_shape = (5,5,1,20)
	W1_init = init_filter(W1_shape, poolsz)
	b1_init = np.zeros(W1_shape[-1], dtype=np.float32)

	W2_shape = (5,5,20,50)
	W2_init = init_filter(W2_shape, poolsz)
	b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

    # vanilla ANN weights
	W3_init = np.random.randn(W2_shape[-1]*7*7, M) / np.sqrt(W2_shape[-1]*7*7 + M)
	b3_init = np.zeros(M, dtype=np.float32)
	W4_init = np.random.randn(M, K) / np.sqrt(M + K)
	b4_init = np.zeros(K, dtype=np.float32)

	X = tf.placeholder(tf.float32, shape=(batch_sz, 28, 28, 1), name='X')
	T = tf.placeholder(tf.float32, shape=(batch_sz, K), name='T')

	# X_ = tf.placeholder(tf.float32, shape=(1000, 28, 28, 1), name='X_')
	# T_ = tf.placeholder(tf.float32, shape=(1000, K), name='T_')

	W1 = tf.Variable(W1_init.astype(np.float32))
 	b1 = tf.Variable(b1_init.astype(np.float32))
 	W2 = tf.Variable(W2_init.astype(np.float32))
	b2 = tf.Variable(b2_init.astype(np.float32))
	W3 = tf.Variable(W3_init.astype(np.float32))
	b3 = tf.Variable(b3_init.astype(np.float32))
	W4 = tf.Variable(W4_init.astype(np.float32))
	b4 = tf.Variable(b4_init.astype(np.float32))

	Z1 = convpool(X, W1, b1)
	Z2 = convpool(Z1, W2, b2)
	Z2_shape = Z2.get_shape().as_list()
	Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
	Z3 = tf.nn.relu( tf.matmul(Z2r, W3) + b3 )
	Yish = tf.matmul(Z3, W4) + b4


	cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(Yish, T))
	train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)
	predict_op = tf.argmax(Yish, 1)
 



	init = tf.initialize_all_variables()
	with tf.Session() as session:
		session.run(init)


		for i in xrange(max_iter):

			for j in xrange(n_batches):

				Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
				Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
				s = session.run(cost, feed_dict = {X: Xbatch, T:Ybatch} );
				test_cost = 0;
				prediction = np.zeros(len(Xtest));

				if len(Xbatch) == batch_sz:
					session.run(train_op, feed_dict = {X: Xbatch, T: Ybatch})
					if j%print_period == 0:
						test_cost = 0;
						prediction = np.zeros(len(Xtest))

						for k in xrange(len(Xtest)/batch_sz):
							Xtestbatch = Xtest[k*batch_sz: (k*batch_sz + batch_sz),]
							Ytestbatch = Ytest_ind[k*batch_sz: (k*batch_sz + batch_sz),]
							test_cost += session.run(cost, feed_dict = {X: Xtestbatch, T:Ytestbatch})

							prediction[k*batch_sz:(k*batch_sz + batch_sz)] = session.run(predict_op, feed_dict={X: Xtestbatch})

						err = error_rate(prediction, Ytest)
						print prediction[1:10]
						print "Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err)


if __name__ == '__main__':
    main()







		










