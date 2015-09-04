__author__ = 'WillCobb'

import random
import time
import numpy as np
import cPickle as pickle
import os


def sayHi():
    print "Hey"

class NeuralNetwork(object):
    def __init__(self, nodes, new=False):
        # nodes is a list containing the number of neurons in each layer
        self.numLayers = len(nodes)
        self.nodes = nodes
        self.newNetwork(nodes)

    def newNetwork(self, nodes):
        self.biases = [np.random.randn(y, 1) for y in nodes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(nodes[:-1], nodes[1:])]

    def saveNetwork(self, fp=None):
        if not (fp):
            fileAppendage = '-'.join([str(x) for x in self.nodes])
            fp = (os.path.dirname(os.path.realpath(__file__)) + "/networkData"+fileAppendage+".pkl")
        pickle.dump([self.biases, self.weights], open(fp, 'w'))

    def loadNetwork(self, fp=None):
        if not (fp):
            fileAppendage = '-'.join([str(x) for x in self.nodes])
            fp = (os.path.dirname(os.path.realpath(__file__)) + "/networkData"+fileAppendage+".pkl")
        self.biases, self.weights = pickle.load(open(fp, 'r'))

    def feedForward(self, a):
        # Return the output of the network if a is input.
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a)+b)
        return a

    def train(self, trainingData, batchSize, eta, test_data, batchTests = True):
        # Train the neural network using each batch.
        # The trainingData is a list of tuples
        # (x, y) represent the training inputs and the desired outputs.
        if test_data: n_test = len(test_data)
        n = len(trainingData)
        random.shuffle(trainingData)
        if (batchTests):
            batches = [trainingData[k:k+batchSize] for k in xrange(0, n, batchSize)] #Seperate data into batches
        else:
            batches = [trainingData,]

        for batch in batches:
            self.updateBatch(batch, eta)

        if (test_data):
            print "Generation Evaluation: {1} / {2}".format(self.score(test_data), n_test)
    def singleTrain(self, trainingData, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases] #initialize an array of 0s to hold data
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        x, y = trainingData[0], trainingData[1]
        delta_nabla_b, delta_nabla_w    = self.backPropagation(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta)*nw  for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta)*nb for b, nb in zip(self.biases, nabla_b)]

    def updateBatch(self, batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases] #initialize an array of 0s to hold data
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            #x is input
            #y is expected result
            delta_nabla_b, delta_nabla_w    = self.backPropagation(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #After testing, update weights and biases
        self.weights = [w-(eta/len(batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backPropagation(self, x, y):
        # Returns (nabla_b, nabla_w) representing the gradient for the cost function C_x.
        # nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar
        # to self.biases and self.weights.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedForward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights): # Runs through each layer and computes the activation gradient
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
        # zs holds all of the activate layers
        # backward pass
        delta = self.costDerivative(activations[-1], y) * sigmoidPrime_vec(zs[-1]) #http://www.holehouse.org/mlclass/09_Neural_Networks_Learning.html
        # activations[-1] is what out neural network thinks the number is
        # delta is how wrong it was
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) #10 x 30 array

        for l in xrange(2, self.numLayers): #l is each layer
            z = zs[-l]
            spv = sigmoidPrime_vec(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
        #After backPropagation, the nablas are used to correct the weights and biases of the network

    def score(self, testData): #Score the data
        results = [(np.argmax(self.feedForward(x)), y)  for (x, y) in testData] #argmax: http://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
        return sum(int(guess == answer) for (guess, answer) in results)
        
    def costDerivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid) # http://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html

def sigmoidPrime(z): #sigmoid Derivative
    return sigmoid(z)*(1-sigmoid(z))

sigmoidPrime_vec = np.vectorize(sigmoidPrime)
