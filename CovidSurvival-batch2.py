import numpy as np
import pandas as pd
import os
import sys
import warnings
from datetime import datetime

from PIL import Image
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, BCEWithLogitsLoss, Sigmoid, BatchNorm1d
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset

#defining the dataset which currently is hardcoded
dataset = pd.read_csv('/Users/arnie/Documents/GitHub/covid-chestxray-dataset/metadata.csv')
df_covid = dataset[dataset.finding == 'COVID-19']
df_covid = df_covid.reset_index()
df_covid.dropna(subset=['survival'], inplace=True)
df_covid = df_covid.reset_index()
df_filenames = df_covid['filename']
imagedir = '/Users/arnie/Documents/GitHub/covid-chestxray-dataset/images/'
os.chdir(imagedir)

#loading and formatting the images
images = []
for i in range(0, len(df_filenames)):
    image = df_filenames[i]
    for root, dirs, files in os.walk(imagedir):
       if image in files:
           images.append(os.path.join(root, image))
       else:
           print(image)

#creating array of images
train_images_raw = []
for i in range(len(images)):
    img = Image.open(images[i]).convert('L')
    img_resize = img.resize((400, 400), Image.ANTIALIAS)
    img_resize_numpy = np.array(img_resize)
    img_resize_numpy = img_resize_numpy.astype(float)
    train_images_raw.append(img_resize_numpy)
#creating array of flipped images
train_images_flipped = []
for i in range(len(images)):
    img_flipped = Image.open(images[i]).convert('L')
    img_flipped_resize = img_flipped.resize((400, 400), Image.ANTIALIAS)
    img_flipped_resize_numpy = np.array(img_flipped_resize)
    img_flipped_resize_numpy = img_flipped_resize_numpy.astype(float)
    img_flipped_actual = np.fliplr(img_flipped_resize_numpy)
    train_images_flipped.append(img_flipped_actual)

#combining the normal and flipped images as well as its labels
train_images = train_images_raw + train_images_flipped
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
train_x = np.array(train_images)
labels1 = df_covid['survival'].values
labels2 = labels1
train_y = np.append(labels1, labels2)

#Plotting images for visualization
plt.figure(figsize=(10,10))
plt.subplot(221), plt.imshow(train_x[1], cmap='gray')
plt.show()

#splitting the dataset into training and validation sets. One can also create a seperate test set here
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.09502262443)
print(train_x.shape, train_y.shape)
print(val_x.shape, val_y.shape)

#converting the training and validation arrays into Tensors
tensor_x = torch.Tensor(train_x)
train_y = train_y.astype(float)
tensor_y = torch.Tensor(train_y)
valtensor_x = torch.Tensor(val_x)
val_y = val_y.astype(float)
valtensor_y = torch.Tensor(val_y)
print(valtensor_x.shape, valtensor_y.shape)

#loading in the training tensors and validation tensors using torch.TensorDataset and Dataloader
training_set = TensorDataset(tensor_x, tensor_y)
validation_set = TensorDataset(valtensor_x, valtensor_y)
training_loader = DataLoader(training_set, batch_size=40, shuffle=True, num_workers=4)
validation_loader = DataLoader(validation_set, batch_size=42, shuffle=True, num_workers=4)

#defining the model architecture
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        #defining convolution layers
        self.cnn_layers = Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        #defining linear layers
        self.linear_layers = Sequential(
            Linear(64 * 100 * 100, 32),
            BatchNorm1d(32),
            ReLU(inplace=True),
            Linear(32, 1),
            #Sigmoid(),
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.00001)
# defining the loss function and weight in case the incoming dataset is skewed
pos_weight = torch.FloatTensor([0.2])
loss = BCEWithLogitsLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    loss = loss.cuda()

print(model)


def train(epoch):
    #creating arrays for later accuracy calculations
    predictionsoverall = []
    labelsoverall = []
    labelsoverallval = []
    predictionsoverallval = []
    model.train()
    tr_loss = 0
    # clearing the Gradients of the model parameters
    optimizer.zero_grad()

    #loading in the training dataset and training over the batch
    for i, (input, labels) in enumerate(training_loader):
        input = input.unsqueeze(1)
        output_train = model(input)
        print(output_train)

        y_train = labels.unsqueeze(1)
        print(y_train)
        #computing training loss
        loss_train = loss(output_train, y_train)
        train_losses.append(loss_train)

        #predictions and reporting accuracy of batch
        predictions = output_train > 0
        labelsoverall += y_train
        print(predictions)
        predictionsoverall += predictions
        #len used to see how many images of the total set have been trained
        print(len(predictionsoverall))
        #calculating accuracy
        accuracy_train = accuracy_score(y_train, predictions)
        print("Accuracy Training This Batch: ", accuracy_train)


        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
        #printing loss per batch
        print('Epoch : ', epoch + 1, '\t', 'loss_train this batch :', loss_train)

    #confusion matrix creation and epoch accuracy calculation
    global cmtrain
    cmtrain = confusion_matrix(labelsoverall, predictionsoverall)
    print(cmtrain)
    accuracy_overall = accuracy_score(labelsoverall, predictionsoverall)
    accuracy_training.append(accuracy_overall)
    print("Accuracy This Epoch: ", accuracy_overall)

    optimizer.zero_grad()
    for i, (inputval, labelsval) in enumerate(validation_loader):
        # prediction for validation set
        inputval = inputval.unsqueeze(1)
        output_val = model(inputval)

        # computing validation loss
        y_val = labelsval.unsqueeze(1)
        loss_val = loss(output_val, y_val)
        val_losses.append(loss_val)

        #predictions for the validation set
        predictions = output_val > 0
        predictionsoverallval += predictions
        labelsoverallval += y_val
        print(len(predictionsoverallval))

        #computing updated weights and printing loss val
        loss_val.backward()
        optimizer.step()
        tr_loss = loss_train.item()
        print('Epoch : ', epoch + 1, '\t', 'loss_val :', loss_val)
        #print('Epoch : ', epoch + 1, '\t', 'loss_val :', loss_val)

    #calculating validation accuracy
    accuracy_overallval = accuracy_score(labelsoverallval, predictionsoverallval)
    accuracy_validation.append(accuracy_overall)
    print("Accuracy Validation This Epoch: ", accuracy_overallval)


# defining the number of epochs
n_epochs = 15
# empty list to store training losses
train_losses = []
# empty list to store validation losses
val_losses = []
# empty lists to store accuracy
accuracy_training = []
accuracy_validation = []
#training the model of epochs
for epoch in range(n_epochs):
    #complining runtime per epoch
    now = datetime.now()
    train(epoch)
    print(epoch+1)
    endnow = datetime.now()
    print(endnow - now)

#plotting loss, accuracy and confusion matrix for training and validation
print(cmtrain)
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()
plt.plot(accuracy_training, label='Accuracy Training')
plt.legend()
plt.show()
plt.plot(accuracy_validation, label='Accuracy Validation')
plt.legend()
plt.show()

plt.matshow(cmtrain, cmap='gray')
plt.show()
