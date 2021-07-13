# Description: Code for Assignment 2 - CNNs
# Course:      Computer Vision
# Authors:     Dimitrios Gagatsis
#              Sacha L. Sindorf
# Date:        2021-05-16
# Description: Makes plots of feature maps

import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

import torch
import torch.nn as nn
import torch.optim as optim

from numpy import load

from architecture import ConvNet0, ConvNet1, ConvNet2, ConvNet3, ConvNet4, ConvNet5



############

# arguments
model_nr = int(sys.argv[1])

# data_dir = './data_small'
data_dir = './data'

# Load data

X_privatetest = load(data_dir+'/X_privatetest.npy')
Y_privatetest = load(data_dir+'/Y_privatetest.npy')

print('--------------------------------------------------------------------------------')
print('Visualize model: ', model_nr)

############

# Normalize
x_test = torch.from_numpy(X_privatetest)/255.0
y_test = torch.from_numpy(Y_privatetest)*1.0

# Evaluate

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

print('model:')
print(model)

# Load trained model
model_name = './models/model_'+str(model_nr)+'.pt'
print('Load trained model: ', model_name)
model.load_state_dict(torch.load(model_name))

model.eval()

# feature map visualization based on:
# https://androidkt.com/how-to-visualize-feature-maps-in-convolutional-neural-networks-using-pytorch/

no_of_layers = 3
conv_layers = [model.layer1, model.layer2, model.layer3]
print('conv_layers:')
print(conv_layers)

# Collect one feature maps for one emotion

labels = list(range(7))
idx = 0
while 0<len(labels):

    # One label from test array
    y_test_cut = y_test[idx]
    y_test_label = np.argmax(y_test_cut).numpy()

    if y_test_label in labels:
        print('Emotion: ', y_test_label)

        # Remove it
        labels.pop(labels.index(y_test_label))

        # One input from test array
        x_test_cut = x_test[idx:idx+1]

        # Feed input through layers
        results = [conv_layers[0](x_test_cut)]
        for i in range(1, len(conv_layers)):
            results.append(conv_layers[i](results[-1]))
        outputs = results

        for num_layer in range(len(outputs)):
            plt.figure(figsize=(10, 10))
            layer_viz = outputs[num_layer][0, :, :, :]
            layer_viz = layer_viz.data
            print('Layer: ',num_layer+1)
            print('layer_viz.shape: ', layer_viz.shape)
            for i, filter in enumerate(layer_viz):
                if i == 64:
                    break
                plt.subplot(8, 8, i + 1)
                plt.imshow(filter, cmap='gray')
                plt.axis("off")
            # plt.show()
            pict = (str(model_nr)+'_'+str(y_test_label)+'_'+str(num_layer) +
                    '_visualize.png')
            plt.savefig(pict, transparent=True, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            print('Save image:',pict )

    idx += 1



