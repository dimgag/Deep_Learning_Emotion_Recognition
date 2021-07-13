# Description: Code for Assignment 2 - CNNs
# Course:      Computer Vision
# Authors:     Dimitrios Gagatsis
#              Sacha L. Sindorf
# Date:        2021-05-16
# Description: - Inspection of the input data
#              - Make histograms

import sys
import numpy as np
import matplotlib.pyplot as plt
import math

from numpy import load

# data_dir = './data_small'
data_dir = './data'

# Y_training = load(data_dir+'/Y_training.npy')
Y_training = load(data_dir+'/Y_privatetest.npy')

Y_training_label = np.argmax(Y_training, axis=1)

labels = range(8)
counts, bins = np.histogram(Y_training_label, bins=labels)

plt.hist(bins[:-1], bins, weights=counts, histtype='bar', align='left', rwidth=0.7)
plt.ylabel('frequency')
plt.xlabel('label')
plt.xticks(labels)

print('bins:')
print(bins)

print('counts:')
print(counts)

plt.show()
