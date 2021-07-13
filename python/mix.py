# Description: Code for Assignment 2 - CNNs
# Course:      Computer Vision
# Authors:     Dimitrios Gagatsis
#              Sacha L. Sindorf
# Date:        2021-05-16
# Description: Mixing in augmented data into training set.

import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn

from numpy import load
from numpy import save

############

# data_dir = './data_small'
data_dir = './data'

print('--------------------------------------------------------------------------------')
print('Mixing training data with augmented data')

############

X_training = load(data_dir+'/X_training.npy')
Y_training = load(data_dir+'/Y_training.npy')

X_collect_flip = load(data_dir+'/X_collect_flip.npy')
Y_collect = load(data_dir+'/Y_collect.npy')

print('len(Y_training): ', len(Y_training))
print('len(X_training): ', len(X_training))
print('len(Y_collect): ', len(Y_collect))

for i0 in range(len(Y_collect)):
    # print('X_collect_flip[i0]: ', X_collect_flip[i0])
    # print('Y_collect[i0]: ', Y_collect[i0])
    rnd = int((np.random.rand(1, 1)[0][0])*len(Y_training))
    # print('rnd: ', rnd )
    X_training = np.insert(X_training, rnd, X_collect_flip[i0], axis=0)
    Y_training = np.insert(Y_training, rnd, Y_collect[i0], axis=0)

print('')
print('len(Y_training): ', len(Y_training))
print('len(X_training): ', len(X_training))

print('Save augmented data')
save('X_training_augmented.npy', X_training)
save('Y_training_augmented.npy', Y_training)
