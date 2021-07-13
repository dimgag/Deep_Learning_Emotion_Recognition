# Description: Code for Assignment 2 - CNNs
# Course:      Computer Vision
# Authors:     Dimitrios Gagatsis
#              Sacha L. Sindorf
# Date:        2021-05-16
# Description: - Check performance of trained model on training data
#              - Make augmented training data

import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn

from numpy import load
from numpy import save

from architecture import ConvNet0, ConvNet1, ConvNet2, ConvNet3, ConvNet4, ConvNet5

############

model_nr = 5

# data_dir = './data_small'
data_dir = './data'

print('--------------------------------------------------------------------------------')
print('Evaluating model on training data: ', model_nr)

############

X_training = load(data_dir+'/X_training.npy')
Y_training = load(data_dir+'/Y_training.npy')

x = torch.from_numpy(X_training)/255.0
y = torch.from_numpy(Y_training)*1.0

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

model_name = './models/model_'+str(model_nr)+'.pt'
print('Load trained model: ', model_name)
model.load_state_dict(torch.load(model_name))

loss_fn = torch.nn.MSELoss(reduction='sum')

model.eval()

# Make a random selection from training set of size test set
# rnd = (np.floor((y.size(0)*np.random.rand(1, 2048  )))).astype(int)
# y_cut = y[rnd]
# x_cut = x[rnd]

y_cut = y
x_cut = x

# Initialize confusion matrix
confusion_matrix = np.zeros((7, 7))

# Collect pictures form confusion matrix
collect_row = 4
collect_column = 6
collect_list = []

with torch.no_grad():
    correct = 0
    total = 0

    # Run model on train data
    y_pred = model(x_cut)

    for i0 in range(y_pred.size(0)):

        # Translate prediction to 'hard' one-hot
        idx = np.argmax(y_pred[i0])
        y_pred_one_hot = np.zeros((1, 7))
        y_pred_one_hot[0][idx] = 1
        y_pred_one_hot = torch.from_numpy(y_pred_one_hot[0].astype(int))*1.0

        # One-hot code to original label
        label = y_cut[i0].tolist().index(1.0)

        # Maintain confusion matrix
        elm = confusion_matrix[label][idx.item()]
        confusion_matrix[label][idx.item()] = 1.0+elm

        # Collect indexes of category in confusion matrix
        if ((collect_row == label) and
            (collect_column == idx.item())):
            collect_list.append(i0)

        # Count for test accuracy
        total += 1
        if torch.equal(y_cut[i0], y_pred_one_hot):
            correct += 1

    print('Test Accuracy: {} %'.format(100 * correct / total))
    print('Confusion matrix:')
    print(confusion_matrix.astype(int))

##################################33

# print('collect_list: ', collect_list)
# print('len(collect_list): ', len(collect_list))

Y_collect = np.zeros((len(collect_list), 7))
X_collect = np.zeros((len(collect_list), 1, 48, 48))

Y_collect = Y_training[collect_list]
X_collect = X_training[collect_list]

# mirror vertically
X_collect_flip = np.flip(X_collect, 3)

print('Save augmented data')
save('X_collect_flip.npy', X_collect_flip)
save('Y_collect.npy', Y_collect)

# Plot for report

fig, axs = plt.subplots(2, 2)

axs[0, 0].imshow(X_collect[0][0]/255.0, cmap='gray')
axs[0, 1].imshow(X_collect_flip[0][0]/255.0, cmap='gray')
axs[0, 0].axis('off')
axs[0, 1].axis('off')

axs[1, 0].imshow(X_collect[1][0]/255.0, cmap='gray')
axs[1, 1].imshow(X_collect_flip[1][0]/255.0, cmap='gray')
axs[1, 0].axis('off')
axs[1, 1].axis('off')

# plt.show()

pict = (str(model_nr) +
        '_augmentation.png')
plt.savefig(pict, transparent=True, bbox_inches='tight', pad_inches=0.1)
plt.close()

print('Save augmentation example: ', pict)

#######################333333


# Make plots

# Take random samples of tests from category in confusion matrix
test_list = []
for i0 in range(4):
    rnd = int(np.floor(np.random.rand(1, 1)*len(collect_list))[0])
    test_idx = collect_list[rnd]
    collect_list.pop(rnd)
    test_list.append(test_idx)

# Fix it for now
# test_list = [1403, 3413, 3588, 2194]
# print('test_list: ', test_list)

image_0 = x[test_list[0]].numpy()[0]
image_1 = x[test_list[1]].numpy()[0]
image_2 = x[test_list[2]].numpy()[0]
image_3 = x[test_list[3]].numpy()[0]

prediction_0 = y_pred[test_list[0]].numpy()
prediction_1 = y_pred[test_list[1]].numpy()
prediction_2 = y_pred[test_list[2]].numpy()
prediction_3 = y_pred[test_list[3]].numpy()

fig, axs = plt.subplots(2, 4)

axs[0, 0].imshow(image_0, cmap='gray')
axs[0, 1].imshow(image_1, cmap='gray')
axs[0, 2].imshow(image_2, cmap='gray')
axs[0, 3].imshow(image_3, cmap='gray')

axs[0, 0].axis('off')
axs[0, 1].axis('off')
axs[0, 2].axis('off')
axs[0, 3].axis('off')

axs[1, 0].bar(range(7), prediction_0)
axs[1, 1].bar(range(7), prediction_1)
axs[1, 2].bar(range(7), prediction_2)
axs[1, 3].bar(range(7), prediction_3)

axs[1, 1].yaxis.set_visible(False)
axs[1, 2].yaxis.set_visible(False)
axs[1, 3].yaxis.set_visible(False)

axs[1, 0].set_xticks(range(7))
axs[1, 1].set_xticks(range(7))
axs[1, 2].set_xticks(range(7))
axs[1, 3].set_xticks(range(7))

axs[1, 0].set_yticks(np.array(range(11))/10.0)
axs[1, 1].sharey(axs[1, 0])
axs[1, 2].sharey(axs[1, 0])
axs[1, 3].sharey(axs[1, 0])

# plt.show()

pict = (str(model_nr)+'_'+str(collect_row)+'_'+str(collect_column) +
        '_confused_training.png')
plt.savefig(pict, transparent=True, bbox_inches='tight', pad_inches=0.1)
plt.close()

print('Save confused images: ', pict)
