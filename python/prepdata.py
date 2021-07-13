# Description: Code for Assignment 2 - CNNs
# Course:      Computer Vision
# Authors:     Dimitrios Gagatsis
#              Sacha L. Sindorf
# Date:        2021-05-16
# Description: - Read fer2013.csv data
#              - Split in training and test
#              - Translate labels to one-hot
#              - Redimension for pytorch convenience
#              - Save as binary numpy


import numpy as np
import csv

from numpy import save

# rid = open('./archive_small/fer2013/fer2013/fer2013.csv', 'r')
rid = open('./archive/fer2013/fer2013/fer2013.csv', 'r')

X_training = np.array([])
Y_training = np.array([])
X_training_cnt = 0

X_publictest = np.array([])
Y_publictest = np.array([])
X_publictest_cnt = 0

X_privatetest = np.array([])
Y_privatetest = np.array([])
X_privatetest_cnt = 0

reader = csv.reader(rid)

skip = True
for ls in reader:
    if skip:
        skip = False
    else:
        image = np.array(ls[1].split()).astype(int)

        if 'Training' == ls[-1]:            
            X_training = np.append(X_training, image)
            Y_training = np.append(Y_training, int(ls[0]))
            X_training_cnt += 1
            
        elif 'PublicTest' == ls[-1]:            
            X_publictest = np.append(X_publictest, image)
            Y_publictest = np.append(Y_publictest, int(ls[0]))
            X_publictest_cnt += 1
            
        elif 'PrivateTest' == ls[-1]:            
            X_privatetest = np.append(X_privatetest, image)
            Y_privatetest = np.append(Y_privatetest, int(ls[0]))
            X_privatetest_cnt += 1

Y_training = Y_training.astype(int)
Y_training_shape = (Y_training.size, Y_training.max()+1)
Y_training_one_hot = np.zeros(Y_training_shape)
Y_training_one_hot[np.arange(Y_training.size), Y_training] = 1

save('X_training.npy', np.resize(X_training, (X_training_cnt, 1, 48, 48)).astype(int))
save('Y_training.npy', Y_training_one_hot.astype(int))

Y_publictest = Y_publictest.astype(int)
Y_publictest_shape = (Y_publictest.size, Y_publictest.max()+1)
Y_publictest_one_hot = np.zeros(Y_publictest_shape)
Y_publictest_one_hot[np.arange(Y_publictest.size), Y_publictest] = 1

save('X_publictest.npy', np.resize(X_publictest, (X_publictest_cnt, 1, 48, 48)).astype(int))
save('Y_publictest.npy', Y_publictest_one_hot.astype(int))

Y_privatetest = Y_privatetest.astype(int)
Y_privatetest_shape = (Y_privatetest.size, Y_privatetest.max()+1)
Y_privatetest_one_hot = np.zeros(Y_privatetest_shape)
Y_privatetest_one_hot[np.arange(Y_privatetest.size), Y_privatetest] = 1

save('X_privatetest.npy', np.resize(X_privatetest, (X_privatetest_cnt, 1, 48, 48)).astype(int))
save('Y_privatetest.npy', Y_privatetest_one_hot.astype(int))



   