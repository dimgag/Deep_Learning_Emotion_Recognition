# Description: Code for Assignment 2 - CNNs
# Course:      Computer Vision
# Authors:     Dimitrios Gagatsis
#              Sacha L. Sindorf
# Date:        2021-05-16
# Description: Evaluate model quality

import sys
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from numpy import load

from architecture import ConvNet0, ConvNet1, ConvNet2, ConvNet3, ConvNet4, ConvNet5

############

# arguments
model_nr = int(sys.argv[1])

# data_dir = './data_small'
data_dir = './data'

# Collect pictures form confusion matrix
collect_row = 4
collect_column = 6
print('--------------------------------------------------------------------------------')
print('Evaluating model: ', model_nr)

############

# Load data
X_training = load(data_dir+'/X_training.npy')
Y_training = load(data_dir+'/Y_training.npy')

x = torch.from_numpy(X_training)/255.0
y = torch.from_numpy(Y_training)*1.0

X_privatetest = load(data_dir+'/X_privatetest.npy')
Y_privatetest = load(data_dir+'/Y_privatetest.npy')

X_publictest = load(data_dir+'/X_publictest.npy')
Y_publictest = load(data_dir+'/Y_publictest.npy')

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

# Load trained model
model_name = './models/model_'+str(model_nr)+'.pt'
print('Load trained model: ', model_name)
model.load_state_dict(torch.load(model_name))

loss_fn = torch.nn.MSELoss(reduction='sum')

model.eval()

# Make a random selection from training set of size test set
rnd = (np.floor((y.size(0)*np.random.rand(1, y_test.size(0))))).astype(int)
y_cut = y[rnd]
x_cut = x[rnd]

# Initialize confusion matrix
confusion_matrix_0 = np.zeros((7, 7))

# Collect pictures from confusion matrix
collect_list = []

with torch.no_grad():
    correct = 0
    total = 0

    # Run model on test data
    y_test_pred = model(x_test)

    # Run model on train data
    # For comparing quality between test and training data
    # Check for overfitting
    y_pred = model(x_cut)

    loss = loss_fn(y_test_pred, y_test)
    loss_train = loss_fn(y_pred, y_cut)
    
    # Loop through model predictions and check quality

    for i0 in range(y_test_pred.size(0)):

        # Translate prediction to 'hard' one-hot
        idx = np.argmax(y_test_pred[i0])
        y_test_pred_one_hot = np.zeros((1, 7))
        y_test_pred_one_hot[0][idx] = 1
        y_test_pred_one_hot = torch.from_numpy(y_test_pred_one_hot[0].astype(int))*1.0

        # One-hot code of test to original label
        label = y_test[i0].tolist().index(1.0)

        # Maintain confusion matrix
        elm = confusion_matrix_0[label][idx.item()]
        confusion_matrix_0[label][idx.item()] = 1.0+elm

        # Collect indexes of category in confusion matrix
        if ((collect_row == label) and
            (collect_column == idx.item())):
            collect_list.append(i0)

        # Count for test accuracy
        total += 1
        if torch.equal(y_test[i0], y_test_pred_one_hot):
            correct += 1

    print('Test Accuracy: {} %'.format(100 * correct / total))
    print('Test Loss:     ', loss)
    print('Training Loss: ', loss_train)
    print('Confusion matrix:')
    print(confusion_matrix_0.astype(int))

# print('collect_list: ', collect_list)
# print('len(collect_list): ', len(collect_list))

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

image_0 = x_test[test_list[0]].numpy()[0]
image_1 = x_test[test_list[1]].numpy()[0]
image_2 = x_test[test_list[2]].numpy()[0]
image_3 = x_test[test_list[3]].numpy()[0]

prediction_0 = y_test_pred[test_list[0]].numpy()
prediction_1 = y_test_pred[test_list[1]].numpy()
prediction_2 = y_test_pred[test_list[2]].numpy()
prediction_3 = y_test_pred[test_list[3]].numpy()

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
        '_confused.png')
plt.savefig(pict, transparent=True, bbox_inches='tight', pad_inches=0.1)
plt.close()

print('Save confused images: ', pict)

# Retrieve standard quality metrics

# Translate to label integers
y_test_pred_label = np.argmax(y_test_pred, axis=1).numpy()
y_test_label = np.argmax(y_test, axis=1).numpy()

confusion_matrix = confusion_matrix(y_test_label, y_test_pred_label)
accuracy_score = accuracy_score(y_test_label, y_test_pred_label)
recall_score = recall_score(y_test_label, y_test_pred_label, average=None)
precision_score = precision_score(y_test_label, y_test_pred_label, average=None)
f1_score = f1_score(y_test_label, y_test_pred_label, average=None)

print('sklearn confusion_matrix:')
print(confusion_matrix)
print('sklearn accuracy_score:')
print(accuracy_score)
print('sklearn recall_score:')
print(recall_score)
print('sklearn precision_score:')
print(precision_score)
print('sklearn f1_score:')
print(f1_score)
