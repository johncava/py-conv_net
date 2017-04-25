from __future__ import division
import numpy as np 

input = np.round(np.random.rand(64,64))
conv1_weight = np.round(np.random.rand(4,4))
conv2_weight = np.round(np.random.rand(2,2))
weights1 = np.random.rand(4,6)
weights2 = np.random.rand(6,6)
weights3 = np.random.rand(6,1)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def conv2d(matrix,n,m,filter,weight):
    convolve = []
    for rS in xrange(0, int(n/filter)):
        c_array = []
        for cS in xrange(0, int(m/filter)):
            conv = matrix[(rS*filter):(rS*filter)+filter,(cS*filter):(cS*filter)+filter]
            conv_ = np.dot(conv,weight)
            sm = np.ndarray.sum(conv_)/(filter ** 2)
            c_array.append(sm)
        convolve.append(c_array)
    convolve = np.array(convolve)
    convolve = sigmoid(convolve)
    return convolve

def max_pool(matrix,n,m,p):
    max_pool = []
    for rS in xrange(0, int(n/p)):
        c_array = np.array([])
        for cS in xrange(0, int(m/p)):
            pool = matrix[(rS*p):(rS*p)+p,(cS*p):(cS*p)+p]
            pool_ = np.amax(pool)
            pool_ = np.array(pool_)
            c_array = np.append(c_array, pool_)
        max_pool.append(c_array)
    max_pool = np.array(max_pool)
    return max_pool

def nn():
    ###
    #Feedforward
    ###
    convolve1 = conv2d(input,64,64,4,conv1_weight)
    print "convolve1\n", convolve1
    pool1 = max_pool(convolve1,16,16,2)
    print "pool1\n",pool1
    convolve2 = conv2d(pool1, 8,8,2, conv2_weight)
    print "convolve2\n",convolve2
    pool2 = max_pool(convolve2, 4,4,2)
    print "pool2\n" ,pool2

    #Fully Connected Neural Network
    fc = pool2.flatten()
    print "fc input layer\n", fc

    hiddenLayer1 = sigmoid(np.dot(fc,weights1))
    print "fc hidden layer 1\n", hiddenLayer1

    hiddenLayer2 = sigmoid(np.dot(hiddenLayer1, weights2))
    print "fc hidden layer 2\n", hiddenLayer2

    output = sigmoid(np.dot(hiddenLayer2, weights3))
    print "output\n", output

    ###
    #Backpropagation
    ###

    #unflatten
    # array.reshape(2,2)


nn()