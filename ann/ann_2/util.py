
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def get_transformed_data():
    print "Reading in and transforming data..."
    df = pd.read_csv('../large_files/train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)

    X = data[:, 1:]
    mu = X.mean(axis=0)
    X = X - mu # center the data
    pca = PCA()
    Z = pca.fit_transform(X)
    Y = data[:, 0]

    return Z, Y, pca, mu

def get_normalized_data():
    print "Reading in and transforming data..."
    df = pd.read_csv('../large_files/train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std # normalize the data
    Y = data[:, 0]
    return X, Y


def forward(X,W,b):
	# softmax
	a = X.dot(W) + b
	expa = np.exp(a);
	y = expa / expa.sum(axis = 1, keepdims = True)
	return y

def predict(p_y):
	return np.argmax(p_y, axis = 1);

def error_rate(p_y, t):
	prediction = predict(p_y)
	return np.mean(prediction != t)


def cost(p_y,t):
	tot = t * np.log(p_y)
	return -tot.sum()

def gradW(t,y,X):
	return X.T.dot(t - y);

def gradb(t,y):
	return (t-y).sum(axis = 0)

def y2indicator(y):
	N = len(y)
	ind = np.zeros((N,10))
	for i in xrange(N):
		ind[i, y[i]] = 1
	return ind

def benchmark_full():
	X, Y = get_normalized_data()

	print "Performing logstic regression..."

	Xtrain = X[:-1000,]
	Ytrain = Y[:-1000]
	Xtest  = X[-1000:,]
	Ytest  = Y[-1000:]

	N, D = Xtrain.shape
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)

	W = np.random.randn(D,10) / 28;
	b = np.zeros(10)
	LL = []
	LLtest = []
	CRtest = []


	# Setting learning rate

	lr = 0.00004;

	reg = 0.01;

	for i in xrange(500):
		p_y = forward(Xtrain, W, b);
		ll = cost(p_y, Ytrain_ind)
		LL.append(ll)

		p_y_test = forward(Xtest, W, b)
		lltest = cost(p_y_test, Ytest_ind)
		LLtest.append(lltest)

		err = error_rate(p_y_test, Ytest)
		CRtest.append(err)

		W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += lr*(gradb(Ytrain_ind, p_y) - reg*b)
        
        if i % 10 == 0:
            print "Cost at iteration %d: %.6f" % (i, ll)
            print "Error rate:", err
 
	p_y = forward(Xtest, W, b)
	print "Final error rate:", error_rate(p_y, Ytest)
	iters = range(len(LL))
	plt.plot(iters, LL, iters, LLtest)
	plt.show()
	plt.plot(CRtest)

	plt.show()


 # if __name__ == '__main__':
 #    # benchmark_pca()
 #    benchmark_full()

















