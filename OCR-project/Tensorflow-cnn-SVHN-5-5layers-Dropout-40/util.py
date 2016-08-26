import pandas as pd;
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf;
import datetime as datetime;

# def get_transformed_data():
# 	print "Reading in and transforming data..."
# 	df_test = pd.read_csv('../large_files/test.csv', header = None)
# 	data_test = df_test.as_matrix().astype(np.float32)
# 	np.random.shuffle(data_train)
# 	np.random.shuffle(data_test)
# 	X_test = data_test[:, 1:]
# 	Y_test = data_test[:,0]
# 	N_test = X_test.shape[0];
# 	X_test = np.reshape(X_test, [ N_test, 32, 32, 3])
# 	X_test  = X_test/255;
# 	return X_test, Y_test,  N_test

def digit(s):
	digit = [];
	s = [int(x) for x in s];
	for j in xrange(len(s)):
		digit.append(str(s[j]))
	Matrix = [[10 for x in range(5)] for y in range(len(s))]
	for j in range(len(s)):
		if len(digit[j]) <= 5:
			for i in range(len(digit[j])):
				Matrix[j][i] = int(digit[j][i]);
		else:
			for i in range(5):
				Matrix[j][i] = int(digit[j][i]);
	return Matrix;

def get_transformed_data():
	print "Reading in and transforming data..."

	df_train = pd.read_csv('train40.csv', header = None)
	df_test = pd.read_csv('test40.csv', header = None)
	data_train = df_train.as_matrix().astype(np.float32)
	data_test = df_test.as_matrix().astype(np.float32)
#	np.random.shuffle(data_train)
#	np.random.shuffle(data_test)

	X_train = data_train[:, 1:]
	Y_train = data_train[:,0]
	X_test = data_test[:, 1:]
	Y_test = data_test[:,0]

	N_train = X_train.shape[0];
	X_train = np.reshape(X_train, [ N_train, 40, 40, 3], order = "F")
	N_test = X_test.shape[0];
	X_test = np.reshape(X_test, [ N_test, 40, 40, 3], order = "F")

	X_train = X_train/255;
	X_test  = X_test/255;
	return X_train, Y_train, X_test, Y_test, N_train, N_test

# We call the first column to be the index.
# ylength2indicator indicates the length of the elements in the index array.

def ytoint(y):
	y = y.astype(int);
	N = len(y);
	lenQ = np.zeros(N)
	for i in xrange(N):
		lenQ[i] = len(str(y[i]));
	return lenQ

def ylength2indicator(y):
	y = y.astype(int);
	N = len(y)
	ind = np.zeros((N,7))
	for i in xrange(N):
		if len(str(y[i])) >= 5:
			ind[i,6] = 1;
		else:	 
			ind[i,len(str(y[i]))] = 1;
	return ind

def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 11))
    for i in xrange(N):
        ind[i, y[i]] = 1
    return ind

def error_rate(p,t):
	return np.mean(p != t);

def init_filter(shape, poolsz):
	w = np.random.randn(*shape)/ np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
	return w.astype(np.float32);
