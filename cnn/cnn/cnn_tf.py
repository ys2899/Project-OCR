
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.signal import convolve2d
from scipy.io import loadmat
from sklearn.utils import shuffle


def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
    for i in xrange(N):
        ind[i, y[i]] = 1
    return ind


def error_rate(p, t):
    return np.mean(p != t)



def convpool(X, W, b):
	conv_out = tf.nn.conv2d(X, W, strides =[1,1,1,1], padding = 'True');
	conv_out = tf.nn.bias_add(conv_out, b)
	pool_out = tf.nn.max_pool(conv_out, ksize = [1,2,2,1], strides =[1,1,1,1], padding = 'True')
	return pool_out


def init_filter(shape, poolsz):
	w = np.random.randn(*shape)/np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
	return w.astype(np.float32);


def rearrange(X):
    # input is (32, 32, 3, N)
    # output is (N, 32, 32, 3)
	N = X.shape[-1]
	out = np.zeros((N, 32, 32, 3), dtype=np.float32)
	for i in xrange(N):
		for j in xrange(3):
			out[i, :, :, j] = X[:, :, j, i]
	return out / 255


def main():

	train = loadmat('../large_files/train_32x32.mat') # N = 73257
	test  = loadmat('../large_files/test_32x32.mat') # N = 26032

    # Need to scale! don't leave as 0..255
    # Y is a N x 1 matrix with values 1..10 (MATLAB indexes by 1)
    # So flatten it and make it 0..9
    # Also need indicator matrix for cost calculation
    Xtrain = rearrange(train['X'])
    Ytrain = train['y'].flatten() - 1
    print len(Ytrain)
    del train
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain_ind = y2indicator(Ytrain)

    Xtest  = rearrange(test['X'])
    Ytest  = test['y'].flatten() - 1
    del test
    Ytest_ind  = y2indicator(Ytest)



