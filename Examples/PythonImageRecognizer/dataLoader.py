import cPickle
import gzip
import os
import numpy as np

def load_data():
    path = os.path.dirname(os.path.realpath(__file__))
    f = gzip.open(path+'/trainingData.pkl.gz', 'rb')
    trainingData, validationData, testData = cPickle.load(f)
    f.close()
    return (trainingData, validationData, testData)

def load_data_wrapper():
    #trainingData, validationData, testingData = [[784 x 1, 1] x 50000]
    #Vectorize results of trainingData for easier comparison to deta in backPropagation
    trainingData, validationData, testingData = load_data()
    trainingInputs = [np.reshape(x, (784, 1)) for x in trainingData[0]]
    trainingResults = [vectorizedResult(y) for y in trainingData[1]]
    trainingData = zip(trainingInputs, trainingResults)
    validationInputs = [np.reshape(x, (784, 1)) for x in validationData[0]]
    validationData2 = zip(validationInputs, validationData[1])
    testInputs = [np.reshape(x, (784, 1)) for x in testingData[0]]
    testData = zip(testInputs, testingData[1])
    return (trainingData, validationData2, testData)

def vectorizedResult(j):
    # Return a 10-dimensional unit vector with a 1.0 in the jthposition and zeroes elsewhere.
    # This is used to convert a digit (0...9) into a corresponding desired output from the neural network.
    vec = np.zeros((10, 1))
    vec[j] = 1.0
    return vec
