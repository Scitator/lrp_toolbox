'''
@author: Kolesnikov Sergey
@maintainer: Kolesnikov Sergey
@contact: sergey.s.kolesnikov@phystech.edu
@date: 16.07.2016
@version: 1.0
'''

import numpy as np
na = np.newaxis

import modules
import model_io
import data_io

# load  MNIST test data and some labels
X = data_io.read('../data/MNIST/test_images.npy')
Y = data_io.read('../data/MNIST/test_labels.npy')

# transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
# X =  X / 127.5 - 1
X =  X / 255.0

# transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
I = Y[:,0].astype(int)
Y = np.zeros([X.shape[0],np.unique(Y).size])
Y[np.arange(Y.shape[0]),I] = 1

#permute data order for demonstration. or not. your choice.
I = np.arange(X.shape[0])
#I = np.random.permutation(I)

# build a network
nn = modules.Sequential(
    [modules.Linear(784, 256),
     modules.Rect(),
     modules.Linear(256, 256),
     modules.Rect(),
     modules.Linear(256, 10),
    #  modules.Tanh(),
    #  modules.Linear(256, 256),
    #  modules.Tanh(),
    #  modules.Linear(256, 10),
     modules.SoftMax()])

# train the network.
nn.train(X,Y, batchsize = 16)

# save the network
model_io.write(nn, '../models/MNIST/mnist_net_small.nn')
