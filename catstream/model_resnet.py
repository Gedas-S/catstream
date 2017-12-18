"""use resnet152 as the model instead of the tutorial one"""
import urllib
import ast
import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.models import resnet152

# I have no idea if I can redistribute the list, so we'll just download it from someone who did :P
CLASSES_LINK = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/c2c91c8e767d04621020c30ed31192724b863041/imagenet1000_clsid_to_human.txt'
CLASSES_LOCATION = 'data/imagenet.classes.txt'

# download class list if none found locally
if not os.path.exists(os.path.abspath(CLASSES_LOCATION)):
    print('No cached image.net classes found, downloading.')
    CLASSES = urllib.request.urlopen(CLASSES_LINK).read()
    if not os.path.exists(os.path.dirname(os.path.abspath(CLASSES_LOCATION))):
        os.makedirs(os.path.dirname(os.path.abspath(CLASSES_LOCATION)))
    with open(CLASSES_LOCATION, 'wb') as file:
        file.write(CLASSES)
    print('Saving image.net classes to %s' % os.path.abspath(CLASSES_LOCATION))
with open(CLASSES_LOCATION) as file:
    CLASSES = ast.literal_eval(file.read())

SIZE_TRANSFORM = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224)
    ])

PREPROCESSING_TRANSFORM = transforms.Compose([
    SIZE_TRANSFORM,
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def train_network():
    """It's pre-trained, do not do anything"""
    get_network()

def get_network():
    """returns the pretrained resnet152 network"""
    return resnet152(pretrained=True).eval().cuda()

def is_cat(category_num):
    """check if this is a cat category"""
    return 280 <= category_num <= 293

def predict_category(image, net):
    """return the predicted category of the image"""
    image = Variable(image.unsqueeze(0).cuda())
    outputs = net(image)
    _, predicted = torch.max(outputs.data, 1) # pylint: disable=no-member

    return predicted[0]
