#!/usr/bin/env python

'''
theta = h * W2 + b2
'''

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    



    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H)) # 10x5
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy)) # 5x10
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    ### Method 1 (using for loop for better understanding)

    hidden = sigmoid(np.dot(data, W1) + b1) # N x H, 20x5
    output = softmax(np.dot(hidden, W2) + b2)
    # print "output", output.shape
    cost = - np.sum(np.log(output) * labels)
    # print "Cost", cost
    ### YOUR CODE HERE: backward propagation
    # theta = h * W2+ b2
    dJ_dtheta = output - labels #
    hidden_grad = sigmoid_grad(hidden)

    gradW1 = np.zeros((10, 5))
    gradb1 = np.zeros((1, 5))
    gradW2 = np.zeros((5, 10))
    gradb2 = np.zeros((1, 10))
     
    delta1 = None
    delta2 = None

    for i in range(data.shape[0]):

        delta2 = W2.dot(dJ_dtheta[i]) 
        delta1 = hidden_grad[i] * delta2

        gradW1 += np.reshape(data[i], (10, 1)).dot(np.reshape(delta1, (1, 5)))
        gradb1 += np.reshape(delta1, (1, 5))
        gradW2 += np.reshape(hidden[i], (5, 1)).dot(np.reshape(dJ_dtheta[i], (1, 10)))
        gradb2 += np.reshape(dJ_dtheta[i], (1, 10))

    ### Method 2 (Only using matrix)
    # dJ_dtheta = output - labels # 20x10
    # hidden_grad = sigmoid_grad(hidden)
    
    # delta2 = dJ_dtheta.dot(W2.T) # 20x5
    # delta1 = np.multiply(hidden_grad, delta2) # 20x5
    
    # gradW1 = np.dot(data.T, delta1)
    # gradb1 = delta1.sum(0)
    # print "gradb1", gradb1.shape
    # gradW2 = np.dot(hidden.T, dJ_dtheta) # 5x10
    # gradb2 = dJ_dtheta.sum(0)
    # print "gradb2", gradb2.shape

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
           gradW2.flatten(), gradb2.flatten()))

    # print "Cost", len(cost)
    # print "Grad", len(grad)
    return cost, grad



def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """ 
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )
    
    # forward_backward_prop(data, labels, params, dimensions)
    gradcheck_naive(lambda params:
          forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
