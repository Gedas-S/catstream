"""define a cnn model and a method to train it"""
# note, pytorch is currently not officially supported on windows, but a kind
# person has made it work
# https://github.com/peterjc123/pytorch-scripts
# get it via
# conda install -c peterjc123 pytorch cuda90
# (assuming your GPU supports cuda90)
#
# also note, this is mostly following the pytorch tutorial at
# http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
#
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from utils import PREPROCESSING_TRANSFORM

MODEL_LOCATION = 'model/catnet.model'

class CatNet(nn.Module):
    """Your typical cnn for image recognition"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, 5) #convolutional layer
        self.pool = nn.MaxPool2d(2, 2) #pooling layer
        self.conv2 = nn.Conv2d(24, 16, 5) #convolutional again
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x): # pylint: disable=arguments-differ
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_network(net=None):
    """Train the CatNet network using CIFAR10 data"""
    # load the data
    trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=PREPROCESSING_TRANSFORM)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # define network, loss function and optimizer
    if net is None:
        net = CatNet()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # stochastic gradient descent

    #train
    for _ in range(2):
        for data in trainloader:
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    if MODEL_LOCATION:
        path = os.path.abspath(MODEL_LOCATION)
        print('Saving model to %s' % path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(net, path)

    return net


def get_network():
    """Get a network - load from disk if available, train othwerise"""
    path = os.path.abspath(MODEL_LOCATION)
    if MODEL_LOCATION and os.path.exists(path):
        print('Loading model from file at %s' % path)
        net = torch.load(path).cuda()
    else:
        print('No saved model found, training a new one.')
        net = train_network()
    return net

def cat(image, net):
    """return the predicted category of the image"""
    image = Variable(image.unsqueeze(0).cuda())
    outputs = net(image)
    _, predicted = torch.max(outputs.data, 1) # pylint: disable=no-member

    return predicted[0]
