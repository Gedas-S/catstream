"""test module for my catstream"""
# pylint: disable=missing-docstring
import unittest
import torch
from PIL import Image
import model_resnet
import utils

class TestStandalone(unittest.TestCase):
    def test_cuda(self):
        self.assertGreater(torch.cuda.device_count(), 0)

    def test_b64_image_encoding(self):
        from base64 import standard_b64encode
        from werkzeug.datastructures import FileStorage
        rdata = FileStorage(open('dot.png', 'rb')) # taken from wiki, public domain
        bimg = standard_b64encode(rdata.read())
        self.assertEqual(bimg, b'iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==')

class TestResnetNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = model_resnet.get_network()

    def test_resnet_image(self):
        # I just drew the image and hereby release it into the public domain :P
        img = Image.open('test_image.jpg')
        img = utils.preprocess_image(img, model_resnet.PREPROCESSING_TRANSFORM)
        cat = utils.category(img, self.net)
        self.assertEqual(type(cat), int)

if __name__ == '__main__':
    unittest.main()
