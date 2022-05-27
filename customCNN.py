# Import required libraries
# general imports
import numpy as np
from numpy import vstack
from numpy import argmax
# import for calculating the accuracy and confusion matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
# import pytorch for neural network
import torch
import torch.nn as nn
from torch import Tensor
# imports for optimizers
from torch.optim import SGD
from torch.optim import Adam
# imports for initializing the weights
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
# import for plotting the metrics
import matplotlib.pyplot as plt


##################### Convolutional Neural Network #####################
# Inspired by https://tinyurl.com/dlmodel
# Adapted for current classification problem
# Define the CNN Model
class CustomCNN(nn.Module):
    # define the model elements
    def __init__(self, conv_input_dim, conv_nfilters, conv_kernel, pool_kernel, fc_hidden_dim, fc_output_dim):
        super(CustomCNN, self).__init__()
        # Input -> First Convolutional layer
        # A 2D Dimension is used because we have image classification
        self.conv_1 = nn.Conv2d(
            in_channels=conv_input_dim, out_channels=conv_nfilters, kernel_size=conv_kernel)
        # Use "He initialization" for the weight
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
        kaiming_uniform_(self.conv_1.weight, nonlinearity='relu')
        # Use ReLU activation function on the first convolutional layer
        self.relu_1 = nn.ReLU()
        # The first pooling layer
        # A 2D Dimension is used because we have image classification
        self.pool_1 = nn.MaxPool2d(kernel_size=pool_kernel, stride=2)
        # First pooling layer -> Second convolutional layer
        self.conv_2 = nn.Conv2d(
            in_channels=conv_nfilters, out_channels=conv_nfilters, kernel_size=conv_kernel)
        # Use "He initialization" for the weight
        kaiming_uniform_(self.conv_2.weight, nonlinearity='relu')
        # Use ReLU activation function on the second hidden layer
        self.relu_2 = nn.ReLU()
        # The second pooling layer
        self.pool_2 = nn.MaxPool2d(kernel_size=pool_kernel, stride=2)
        # Second pooling layer -> The fully connected layer
        self.hidden_1 = nn.Linear(conv_nfilters*25, fc_hidden_dim)
        # Use "He initialization" for the weight
        kaiming_uniform_(self.hidden_1.weight, nonlinearity='relu')
        # Use ReLU activation function on the first hidden layer
        self.relu_3 = nn.ReLU()
        # Third hidden layer -> Second hidden layer -> Output layer
        self.hidden_2 = nn.Linear(fc_hidden_dim, fc_output_dim)
        # Use the "Glorot initialization" for the weight
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
        xavier_uniform_(self.hidden_2.weight)
        # Use Softmax activation function on the output, since this is a multiclass problem
        self.softmax = nn.Softmax(dim=1)

    # Forward propagate
    def forward(self, x):
        # Input -> First hidden layer
        x = self.conv_1(x)
        # Activate ReLU
        x = self.relu_1(x)
        # Pool results
        x = self.pool_1(x)
        # Pool layer -> Second hidden layer
        x = self.conv_2(x)
        # Activate ReLU
        x = self.relu_2(x)
        # Pool results
        x = self.pool_2(x)
        # Flatten image -> Transform pooled feature map to a vector
        #x = x.view(-1, 5*5*32)
        x = x.reshape(x.size(0), -1)
        # Third hidden layer
        x = self.hidden_1(x)
        # Activate ReLU
        x = self.relu_3(x)
        # Third hidden layer -> Fourth hidden layer -> Output layer
        x = self.hidden_2(x)
        # Apply Softmax
        x = self.softmax(x)
        return x


# Function to train the model on the training set and evaluate on the validation set
# Inputs: The training dataloader, the validation dataloader, the model to train,
# the criterion (loss) calculator, the optimizer, learning rate, number of epochs,
# counter and required lists for evaluation
# Outputs: Loss and accuracy on the validation set
def train_val_CNN(train_dataloader, val_dataloader, model, criterion, optimizer, learning_rate, n_epochs, count, loss_lst, iteration_lst, accuracy_lst):
    # Select optimizer based on input
    if optimizer == "SGD":
        # set term of momentum
        momentum = 0.9
        # select Stohastic Gradient Descent
        optimizer = SGD(model.parameters(),
                        lr=learning_rate, momentum=momentum)
    else:
        # select Adam
        optimizer = Adam(model.parameters(), lr=learning_rate)
    # Enumerate epochs
    for epoch in range(n_epochs):
        model.train()
        # Enumerate batches
        for i, (images, labels) in enumerate(train_dataloader):
            # Clear the gradients
            optimizer.zero_grad()
            # Compute the output
            output = model(images)
            # Calculate the SoftMax and CE loss
            loss = criterion(output, labels)
            # Calculate the gradients
            loss.backward()
            # Update the weights
            optimizer.step()

            # Code to evaluate model on the validation set
            count += 1
            # Control when to calculate accuracy
            if count % 50 == 0:
                # Accuracy calculation
                correct = 0
                total = 0
                model.eval()
                # Make predictions on the validation set
                for images_val, labels_val in val_dataloader:
                    # Compute the output
                    output_val = model(images_val)
                    # Get predictions from the maximum value
                    predicted_val = torch.max(output_val.data, 1)[1]
                    # Get the total number of labels
                    total += len(labels_val)
                    # Get the total correct predictions
                    correct += (predicted_val == labels_val).sum()
                # Calculate Accuracy
                accuracy = 100 * correct / float(total)
                # Store the loss and iteration number
                loss_lst.append(loss.data)
                iteration_lst.append(count)
                accuracy_lst.append(accuracy)
                # Print the loss
                if count % 500 == 0:
                    print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(
                        count, loss.data, accuracy))
    # Visualize the loss and accuracy in the end
    # Loss
    plt.plot(iteration_lst, loss_lst)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("CNN: Loss vs Number of iteration")
    plt.show()
    # Accuracy
    plt.plot(iteration_lst, accuracy_lst, color="red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.title("CNN: Accuracy vs Number of iteration")
    plt.show()


# Function to train the model on the whole training set
# Inputs: The training dataloader, the model to train,
# the criterion (loss) calculator, the optimizer, learning rate, number of epochs,
def train_final_CNN(ftrain_dataloader, model, criterion, optimizer, learning_rate, n_epochs):
    # Select optimizer based on input
    if optimizer == "SGD":
        # set term of momentum
        momentum = 0.9
        # select Stohastic Gradient Descent
        optimizer = SGD(model.parameters(),
                        lr=learning_rate, momentum=momentum)
    else:
        # select Adam
        optimizer = Adam(model.parameters(), lr=learning_rate)
    # Enumerate epochs
    for epoch in range(n_epochs):
        model.train()
        # Enumerate batches
        for i, (images, labels) in enumerate(ftrain_dataloader):
            # Clear the gradients
            optimizer.zero_grad()
            # Compute the output
            output = model(images)
            # Calculate the SoftMax and CE loss
            loss = criterion(output, labels)
            # Calculate the gradients
            loss.backward()
            # Update the weights
            optimizer.step()


# Function to test the model on the final test set
# Inputs: The final test dataloader, the optimized model
# Output: The accuracy of the optimized model on the final test set
def test_CNN(test_dataloader, model):
    # Create prediction list
    pred_lst = list()
    # Create list with labels
    labels_lst = list()
    model.eval()
    # Enumerate batches
    for i, (images, labels) in enumerate(test_dataloader):
        # Evaluate the model on the validation set
        output = model(images)
        # Get the numpy arrays
        output = output.detach().numpy()
        actual = labels.numpy()
        # Use argmax to convert to labels
        output = argmax(output, axis=1)
        # Reshape
        output = output.reshape((len(output), 1))
        actual = actual.reshape((len(actual), 1))
        # Append to lists
        pred_lst.append(output)
        labels_lst.append(actual)
    # Vertical stack
    pred_lst = vstack(pred_lst)
    labels_lst = vstack(labels_lst)
    # Calculate Accuracy
    accuracy = accuracy_score(labels_lst, pred_lst)
    accuracy = (np.round(accuracy, decimals=3) * 100).astype(int)
    # Calculate Confusion Matrix
    conmat = confusion_matrix(labels_lst, pred_lst)
    # Create confusion matrix display object
    disp = ConfusionMatrixDisplay(confusion_matrix=conmat)
    # Print Accuracy and F1 Score
    print('Accuracy: {}%'.format(accuracy))
    # Show the confusion matrix
    disp.plot()
    plt.title('CNN - Confusion Matrix')
    plt.show()


# Function to predict the digit of a given image
# Inputs: A tensor image of a number, the optimized model
# Output: The label of the predicted number
def predict_digit_CNN(image, model):
    # convert the row to a list
    image = image.tolist()
    # convert the row to tensor data
    image = Tensor([image])
    # Make the prediction
    output = model(image)
    # Get the numpy array
    output = output.detach().numpy()
    # Get the label of the number
    output = argmax(output)
    # Print the number
    print('Predicted the number {}'.format(output))
