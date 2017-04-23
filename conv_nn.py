from __future__ import division
import numpy as np 

input = np.round(np.random.rand(8,8))
conv1_weight = np.round(np.random.rand(3,3))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def conv2d(matrix,n,m):
    convolve = []
    for rS in xrange(0, n - 2):
        c_array = []
        for cS in xrange(0, m - 2):
            conv = matrix[rS:rS+3,cS:cS+3]
            conv_ = np.dot(conv,conv1_weight)
            sm = np.ndarray.sum(conv_)
            c_array.append(sm)
        convolve.append(c_array)
    convolve = np.matrix(convolve)
    convolve = sigmoid(convolve)
    return convolve

def max_pool(matrix,n,m):
    max_pool = []
    for rS in xrange(0, n - 1):
        c_array = []
        for cS in xrange(0,m - 1):
            pool = matrix[rS:rS+2,cS:cS+2]
            pool_ = np.amax(pool)
            c_array.append(pool_)
        max_pool.append(c_array)
    max_pool = np.matrix(max_pool)
    return max_pool

def nn():
    convolve1 = conv2d(input,8,8)
    print convolve1
    pool1 = max_pool(convolve1,6,6)
    print pool

nn()