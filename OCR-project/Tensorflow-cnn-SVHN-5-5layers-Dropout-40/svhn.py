import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

import tensorflow as tf;
import pandas as pd;
import numpy as np
import matplotlib.pyplot as plt
from util import get_transformed_data, error_rate, digit
from util import y2indicator;
from util import init_filter;
from util import ytoint;
from util import ylength2indicator;
from datetime import datetime;

model_path = "model.ckpt";
keep_prob = tf.placeholder(tf.float32);
keep_prob0 = tf.placeholder(tf.float32);
batch_sz = 2000;

def conv1pool(X,W,b):	
	conv_out = tf.nn.conv2d(X,W,strides = [1,1,1,1], padding= 'SAME');
	conv_out = tf.nn.bias_add(conv_out, b);
	conv1pool = tf.nn.relu(conv_out);
	conv1pool = tf.nn.dropout(conv1pool, keep_prob0);
	pool_out = tf.nn.max_pool(conv1pool, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	return pool_out;

def conv2pool(X,W,b):
	conv_out = tf.nn.conv2d(X,W,strides = [1,1,1,1], padding= 'SAME');
	conv_out = tf.nn.bias_add(conv_out, b);
	conv2pool = tf.nn.relu(conv_out);
	conv2pool = tf.nn.dropout(conv2pool, keep_prob);
	pool_out = tf.nn.max_pool(conv2pool, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	return pool_out;

def main():
	X_train, Y_train, X_test, Y_test, N_train, N_test = get_transformed_data();
	X_train = X_train[:32000,]
	Y_train = Y_train[:32000]
	X_test = X_test[:12000,]
	Y_test = Y_test[:12000]
	Y_test_for_comp = Y_test;
	Y_train_ind = ylength2indicator(Y_train)
	Y_test_ind = ylength2indicator(Y_test);
	# Above is to compute the length indicator;
	Y_train_Q = ytoint(Y_train);
	Y_test_Q = ytoint(Y_test);
	# Above is to compute the length only;

	## About the iterations
	max_iter = 150;
	print_period = 1000;
	N = X_train.shape[0]
	n_batches = N/ batch_sz
	M = 2048;
	K = [7, 11];
	KM = 3072;
	poolsz = (2,2);

	##Placeholder and other variables.
	X = tf.placeholder(tf.float32, shape=(batch_sz, 40, 40, 3), name='X')
	T0 = tf.placeholder(tf.float32, shape=(batch_sz, K[0]), name='T')
	T1 = tf.placeholder(tf.float32, shape=(batch_sz, K[1]), name='T')

	## About Optimization parameters. 
	W1_shape = (5,5,3,16)  #(filter_width, filter_height, num_col_chanels, num_feature_maps)
	W1_init = init_filter(W1_shape, poolsz)
	b1_init = np.zeros(W1_shape[-1], dtype = np.float32)
	W2_shape = (5, 5, 16, 32) # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
	W2_init = init_filter(W2_shape, poolsz)
	b2_init = np.zeros(W2_shape[-1], dtype=np.float32)
	W3_shape = (5, 5, 32, 48) # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
	W3_init = init_filter(W3_shape, poolsz)
	b3_init = np.zeros(W3_shape[-1], dtype=np.float32)
	W4_shape = (3, 3, 48, 64) # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)]
	W4_init = init_filter(W4_shape, poolsz)
	b4_init = np.zeros(W4_shape[-1], dtype=np.float32)
	W5_shape = (3, 3, 64, 128) # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)]
	W5_init = init_filter(W5_shape, poolsz)
	b5_init = np.zeros(W5_shape[-1], dtype=np.float32)




	W6_init = np.random.randn(W5_shape[-1]*2*2, M) / np.sqrt(W4_shape[-1]*2*2 + M)
	b6_init = np.zeros(M, dtype=np.float32)

	W7_init = np.random.randn(M, KM) / np.sqrt(M + KM)
	b7_init = np.zeros(KM, dtype=np.float32)
	W8_init = np.random.randn(KM, K[0]) / np.sqrt(KM + K[0])
	b8_init = np.zeros(K[0], dtype=np.float32)
	W8N_init = np.random.randn(KM, K[1]) / np.sqrt(KM + K[1])
	b8N_init = np.zeros(K[1], dtype=np.float32)


	W1_L = tf.Variable(W1_init.astype(np.float32))
	b1_L = tf.Variable(b1_init.astype(np.float32))
	W2_L = tf.Variable(W2_init.astype(np.float32))
	b2_L = tf.Variable(b2_init.astype(np.float32))
	W3_L = tf.Variable(W3_init.astype(np.float32))
	b3_L = tf.Variable(b3_init.astype(np.float32))
	W4_L = tf.Variable(W4_init.astype(np.float32))
	b4_L = tf.Variable(b4_init.astype(np.float32))	
	W5_L = tf.Variable(W5_init.astype(np.float32))
	b5_L = tf.Variable(b5_init.astype(np.float32))
	W6_L = tf.Variable(W6_init.astype(np.float32))
	b6_L = tf.Variable(b6_init.astype(np.float32))
	W7_L = tf.Variable(W7_init.astype(np.float32))
	b7_L = tf.Variable(b7_init.astype(np.float32))
	W8_L = tf.Variable(W8_init.astype(np.float32))
	b8_L = tf.Variable(b8_init.astype(np.float32))


	Z1_L = conv1pool(X,W1_L,b1_L);
	Z2_L = conv2pool(Z1_L, W2_L, b2_L)
	Z3_L = conv2pool(Z2_L, W3_L, b3_L)
	Z4_L = conv2pool(Z3_L, W4_L, b4_L)

	print Z4_L;

	Z5_L = conv2pool(Z4_L, W5_L, b5_L);

	print Z5_L; 



	Z5_shape_L = Z5_L.get_shape().as_list()
	Z5r_L = tf.reshape(Z5_L, [Z5_shape_L[0], np.prod(Z5_shape_L[1:])])
	Z5r_L = tf.nn.dropout(Z5r_L, keep_prob)

	print Z5r_L;


	Z6_L = tf.nn.relu( tf.matmul(Z5r_L, W6_L) + b6_L);
	Z6_L = tf.nn.dropout(Z6_L, keep_prob)

	print Z6_L;

	Z7_L = tf.nn.relu( tf.matmul(Z6_L, W7_L) + b7_L);
	Z7_L = tf.nn.dropout(Z7_L, keep_prob);


	Yish_L = tf.matmul(Z7_L, W8_L) + b8_L;
	cost_L = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(Yish_L, T0))	
	train_op_L = tf.train.RMSPropOptimizer(0.0001, decay=0.98, momentum=0.9).minimize(cost_L)	
	predict_op_L = tf.argmax(Yish_L, 1)


	W1 = [0,0,0,0,0];
	b1 = [0,0,0,0,0];
	W2 = [0,0,0,0,0];
	b2 = [0,0,0,0,0];
	W3 = [0,0,0,0,0];
	b3 = [0,0,0,0,0];
	W4 = [0,0,0,0,0];
	b4 = [0,0,0,0,0];
	W5 = [0,0,0,0,0];
	b5 = [0,0,0,0,0];
	W6 = [0,0,0,0,0];
	b6 = [0,0,0,0,0];
	W7 = [0,0,0,0,0];
	b7 = [0,0,0,0,0];	
	W8 = [0,0,0,0,0];
	b8 = [0,0,0,0,0];	
	Yish = [0,0,0,0,0];
	cost = [0,0,0,0,0];
	train_op = [0,0,0,0,0];
	predict_op = [0,0,0,0,0];


	for h in range(5):
		W1[h] = tf.Variable(W1_init.astype(np.float32))
		b1[h] = tf.Variable(b1_init.astype(np.float32))
		W2[h] = tf.Variable(W2_init.astype(np.float32))
		b2[h] = tf.Variable(b2_init.astype(np.float32))
		W3[h] = tf.Variable(W3_init.astype(np.float32))
		b3[h] = tf.Variable(b3_init.astype(np.float32))
		W4[h] = tf.Variable(W4_init.astype(np.float32))
		b4[h] = tf.Variable(b4_init.astype(np.float32))
		W5[h] = tf.Variable(W5_init.astype(np.float32))
		b5[h] = tf.Variable(b5_init.astype(np.float32))
		W6[h] = tf.Variable(W6_init.astype(np.float32))
		b6[h] = tf.Variable(b6_init.astype(np.float32))
		W7[h] = tf.Variable(W7_init.astype(np.float32))
		b7[h] = tf.Variable(b7_init.astype(np.float32))
		W8[h] = tf.Variable(W8N_init.astype(np.float32))
		b8[h] = tf.Variable(b8N_init.astype(np.float32))

		Z1 = conv1pool(X,W1[h],b1[h]);
		Z2 = conv2pool(Z1, W2[h], b2[h]);
		Z3 = conv2pool(Z2, W3[h], b3[h]);
		Z4 = conv2pool(Z3, W4[h], b4[h]);
		Z5 = conv2pool(Z4, W5[h], b5[h]);
		Z5_shape = Z5.get_shape().as_list()
		Z5r = tf.reshape(Z5, [Z5_shape[0], np.prod(Z5_shape[1:])])
		Z5r = tf.nn.dropout(Z5r, keep_prob)
		Z6 = tf.nn.relu( tf.matmul(Z5r, W6[h]) + b6[h]);
		Z6 = tf.nn.dropout(Z6, keep_prob)
		Z7 = tf.nn.relu( tf.matmul(Z6, W7[h]) + b7[h]);
		Z7 = tf.nn.dropout(Z7, keep_prob);


		Yish[h] = tf.matmul(Z7, W8[h]) + b8[h];
		cost[h] = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(Yish[h], T1))	
		train_op[h] = tf.train.RMSPropOptimizer(0.0001, decay=0.98, momentum=0.9).minimize(cost[h])	
		predict_op[h] = tf.argmax(Yish[h], 1)


## Save all the variables.
	saver = tf.train.Saver();
	LL = [];
	Error_Testing = [];
	Yish_log = []; 
	init = tf.initialize_all_variables();

	with tf.Session() as session:
		session.run(init);
		print 'Computing the length of the number.'
		for i in xrange(max_iter):
			Yish_ = np.zeros((len(X_test), K[0]));
			for j in xrange(n_batches):		
				Xbatch = X_train[j*batch_sz:(j*batch_sz + batch_sz),]
				Ybatch = Y_train_ind[j*batch_sz:(j*batch_sz + batch_sz),]
				if len(Xbatch) == batch_sz:
					session.run(train_op_L, feed_dict = {X: Xbatch, T0: Ybatch, keep_prob0:0.8, keep_prob: 0.5})
					if j%print_period == 0:
						test_cost = 0;
						prediction_test = np.zeros(len(X_test));
						prediction_train = np.zeros(len(X_train));
						
						for k in xrange(len(X_test)/batch_sz):
							Xtestbatch = X_test[k*batch_sz: (k*batch_sz + batch_sz),]
							Ytestbatch = Y_test_ind[k*batch_sz: (k*batch_sz + batch_sz),]
							test_cost += session.run(cost_L, feed_dict = {X: Xtestbatch, T0:Ytestbatch, keep_prob0:1, keep_prob: 1})
							prediction_test[k*batch_sz:(k*batch_sz + batch_sz)] = session.run(predict_op_L, feed_dict={X: Xtestbatch,keep_prob0:1, keep_prob:1})
							Yish_[k*batch_sz:(k*batch_sz + batch_sz)] = session.run(Yish_L, feed_dict = {X:Xtestbatch, keep_prob0:1, keep_prob:1});
						err_testing = error_rate(prediction_test, Y_test_Q)
						print "Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err_testing)
						LL.append(test_cost)
						Error_Testing.append(err_testing);		
		Yish_ = session.run(tf.nn.log_softmax(Yish_));
		Yish_log = np.array([Yish_]);	
		Yish_log_length = Yish_log;
		Yish_log_length = np.reshape(Yish_log_length, (len(X_test), 1, K[0]))

# # # # 	# Evaluating the digit.
		Y_train_ = np.array(digit(Y_train));
		Y_test_ =  np.array(digit(Y_test));
		Yish_log_ = np.array([]).reshape(len(X_test),2,0);

		for h in range(5):
			Y_train = Y_train_[:,h];
			Y_test = Y_test_[:,h];
			Y_train_ind = y2indicator(Y_train);
			Y_test_ind = y2indicator(Y_test);

			t0 = datetime.now()
			LL = []
			Error_Training = [];
			Error_Testing = [];
			Yish_log = []; 
			
			print 'Computing the %d digit of the number.'% (h)
			for i in xrange(max_iter):
				for j in xrange(n_batches):
					Xbatch = X_train[j*batch_sz:(j*batch_sz + batch_sz),]
					Ybatch = Y_train_ind[j*batch_sz:(j*batch_sz + batch_sz),]
					if len(Xbatch) == batch_sz:
						session.run(train_op[h], feed_dict = {X: Xbatch, T1: Ybatch,keep_prob0:0.8, keep_prob:0.5})
						if j%print_period == 0:
							test_cost = 0;
							prediction_test = np.zeros(len(X_test));
							prediction_train = np.zeros(len(X_train));
							Yish_ = np.zeros((len(X_test), K[1])); 
							
							for k in xrange(len(X_test)/batch_sz):
								Xtestbatch = X_test[k*batch_sz: (k*batch_sz + batch_sz),]
								Ytestbatch = Y_test_ind[k*batch_sz: (k*batch_sz + batch_sz),]
								test_cost += session.run(cost[h], feed_dict = {X: Xtestbatch, T1:Ytestbatch, keep_prob0:1, keep_prob:1})
								prediction_test_batch = session.run(predict_op[h], feed_dict={X: Xtestbatch, keep_prob0:1,  keep_prob:1})							
								for n,item in enumerate(prediction_test_batch):
									if item==10:
										prediction_test_batch[n] = 0;							
								prediction_test[k*batch_sz:(k*batch_sz + batch_sz)] = prediction_test_batch;
								Yish_[k*batch_sz:(k*batch_sz + batch_sz)] = session.run(Yish[h], feed_dict = {X:Xtestbatch, keep_prob0:1,  keep_prob:1});

							Y_test_transformed = Y_test;	
							for n,item in enumerate(Y_test):
								if item==10:
									Y_test_transformed[n] = 0;	
							for n,item in enumerate(Yish_):
								if np.argmax(item, axis = 0) == 10:
									Yish_[n] = [0.0909]*11;						

							err_testing = error_rate(prediction_test, Y_test_transformed)
							print "Cost / err on digit h=%d at iteration i=%d, j=%d: %.3f / %.3f" % (h, i, j, test_cost, err_testing)			
							LL.append(test_cost)
							Error_Testing.append(err_testing);	
				print(session.run(W1[h][0,0,0,3]))

			Yish_ = session.run(tf.nn.log_softmax(Yish_));
			for itr in range(len(Yish_)):
				Yish_log.append([prediction_test[itr], Yish_[itr, int(prediction_test[itr])]]);
			Yish_log = np.array(Yish_log);	
			Yish_log_ = np.dstack((Yish_log_, Yish_log));				
			# print np.shape(Yish_log_);
			save_path = saver.save(session, model_path)

	# To make an artificial form.	
	b = np.zeros((len(X_test), 2, 1))
	Yish_log_ = np.concatenate((b,Yish_log_), axis = 2);
	Yish_log_ = np.concatenate((Yish_log_, b), axis = 2);		

	Yish_log_whole = np.concatenate((Yish_log_, Yish_log_length), axis =1);
	# print np.shape(Yish_log_whole);
	# print Yish_log_whole[1:2,]
	# Inference of the whole number#############
	# Argmax statistics.
	Inf_digit = np.zeros((len(X_test), 7));
	Inf_num = np.zeros(len(X_test))
	Inf_digit[:,0] = Yish_log_whole[:,1,0] + Yish_log_whole[:,2,0];

	for j in range(len(X_test)):
		for i in range(1,7):
			Inf_digit[j,i] = sum(Yish_log_whole[j,1,1:i]) + Yish_log_whole[j,2,i];

	Length_digit = np.argmax(Inf_digit,1);
	# Inference
	for i in range(len(Length_digit)):
		if Length_digit[i] == 0:
			Inf_num[i] = 0;
		else:
			Inf_num[i] = ''.join([str(int(x)) for x in Yish_log_whole[i,0, 1:Length_digit[i] + 1]]);
	
	Inf_num = [int(x) for x in Inf_num]
	print Inf_num[0:9];
	print Y_test_for_comp[0:9];	
	#Evaluation of the Error rate:
	err_testing = error_rate(Y_test_for_comp, Inf_num)
	print err_testing;

	#### Restoring my model Test

if __name__ == "__main__":
    main()
