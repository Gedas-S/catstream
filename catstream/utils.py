"""functions to assess a network"""
import torch
from torch.autograd import Variable

def category(image, net):
    """return the predicted category of the image"""
    image = Variable(image.unsqueeze(0).cuda())
    outputs = net(image)
    _, predicted = torch.max(outputs.data, 1) # pylint: disable=no-member

    return predicted[0]
