"""functions to assess a network"""
import matplotlib.pyplot as pyplot
import numpy
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
from model import PREPROCESSING_TRANSFORM

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def show_image(img):
    """show an image"""
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    pyplot.imshow(numpy.transpose(npimg, (1, 2, 0)))

def show_images(loader, net, classes=CIFAR10_CLASSES):
    """show the first batch of images from the loader"""
    dataiter = iter(loader)
    images, labels = dataiter.next()
    labels = labels.cuda()

    show_image(torchvision.utils.make_grid(images))
    print('Truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1) # pylint: disable=no-member

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))

def calculate_accuracy(loader, net):
    """calculated the accuracy of a network using the provided loader as the testloader"""
    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        labels = labels.cuda()
        outputs = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1) # pylint: disable=no-member
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return correct / total

def print_accuracy(accuracy):
    """print nicely formated accuracy"""
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * accuracy))

def cifar10_testdata():
    """load the test part of cifar10 data and return it"""
    testset = datasets.CIFAR10(root='./data', train=False, download=True,
                               transform=PREPROCESSING_TRANSFORM)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    return testloader

def preprocess_image(img):
    """preprocesses an image using the defined preprocessing transforms"""
    return PREPROCESSING_TRANSFORM(img)

def cat(image, net):
    """return the predicted category of the image"""
    image = Variable(image.unsqueeze(0).cuda())
    outputs = net(image)
    _, predicted = torch.max(outputs.data, 1) # pylint: disable=no-member

    return predicted[0]
