# Description: Code for Assignment 2 - CNNs
# Course:      Computer Vision
# Authors:     Dimitrios Gagatsis
#              Sacha L. Sindorf
# Date:        2021-05-16
# Description: Pytorch training


import sys
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn

from numpy import load
from torch import optim

from architecture import ConvNet0, ConvNet1, ConvNet2, ConvNet3, ConvNet4, ConvNet5

# Using GPU to accelerate training speed
gpu_present = torch.cuda.is_available()

if gpu_present:
    print('CUDA')
    dev = "cuda:0"
else:
    print('NO CUDA')
    dev = "cpu"

device = torch.device(dev)

# net = ConvNet()
# print(net)

############

# arguments
model_nr = int(sys.argv[1])
nr_epochs = int(sys.argv[2])

# data_dir = './data_small'
data_dir = './data'

# batch_size = 32
# batch_size = 64
batch_size = 128

print('--------------------------------------------------------------------------------')
print('Training model: ', model_nr)
print('Number of epochs: ', nr_epochs)

############

# Load training data
X_training = load(data_dir+'/X_training.npy')
Y_training = load(data_dir+'/Y_training.npy')

learning_rate = 1e-4

if 0 == model_nr:
    model = ConvNet0()
elif 1 == model_nr: 
    model = ConvNet1()
elif 2 == model_nr: 
    model = ConvNet2()
elif 3 == model_nr: 
    model = ConvNet3()
elif 4 == model_nr: 
    model = ConvNet4()
elif 5 == model_nr: 
    model = ConvNet5()
elif 6 == model_nr: 
    model = ConvNet5()
    # Same architecture but different training data
    X_training = load(data_dir+'/X_training_augmented.npy')
    Y_training = load(data_dir+'/Y_training_augmented.npy')

# Normalize
x = torch.from_numpy(X_training)/255.0
y = torch.from_numpy(Y_training)*1.0

# Loss function and optimizer
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if gpu_present:
    model = model.cuda()
    loss_fn = loss_fn.cuda()

#print('x: ', x)

print('x shape: ', x.shape)
#print('y shape: ', y.shape)
#print('y_pred shape: ', y_pred.shape)

# Number of batches
nr_batches = math.ceil(x.shape[0]/batch_size)
#print('nr_batches: ', nr_batches)

loss_list = []
for epoch in range(nr_epochs):
    for i0 in range(nr_batches):
    
        # indexes for mini-batch
        left = i0*batch_size
        right = min(left+batch_size, x.shape[0])

        x_batch = x[left:right]
        y_batch = y[left:right]

        if gpu_present:
            x_batch = x_batch.to('cuda', non_blocking=True)
            y_batch = y_batch.to('cuda', non_blocking=True)

        y_pred = model(x_batch)         # compute the prediction
        loss = loss_fn(y_pred, y_batch) # compute the difference between prediction and ground truth
        optimizer.zero_grad()     # zero the gradients before running the backward pass
        loss.backward()           # perform a backward pass
        optimizer.step()          # update the parameters

    if epoch % 10 == 9:
        print('epoch:', epoch)
        print('loss: ', loss.item())
        loss_list.append(loss.item())

# Save model
model_name = './model_'+str(model_nr)+'.pt'
print('Save trained model: ', model_name)
torch.save(model.state_dict(), model_name)

# Plot devepment of loss over epochs
plt.plot(loss_list)
plt.ylabel("Loss")
plt.xlabel("Epoch/10")
plt.show()
