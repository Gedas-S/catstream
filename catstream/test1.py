"""test module for my catstream"""
# pylint: disable=missing-docstring
import unittest
import torch
from PIL import Image
import model
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

class TestNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = model.get_network()

    @unittest.skip("don't want to run a 1.5 minute test every time")
    def test_model_train(self):
        self.net = model.train_network()
        self.test_model_accuracy()

    def test_model_accuracy(self):
        acc = utils.calculate_accuracy(utils.cifar10_testdata(), self.net)
        utils.print_accuracy(acc)
        self.assertGreater(acc, 0.5) # note the expected random guessing accuracy would be 0.1,
                                     # so we are quite demanding here

    # @unittest.skip('cannot see matplotlib images in VS anyway')
    def test_images(self):
        utils.show_images(utils.cifar10_testdata(), self.net)

    def test_image(self):
        # I just drew the image and hereby release it into the public domain :P
        img = Image.open('test_image.jpg')
        img = utils.preprocess_image(img)
        cat = utils.cat(img, self.net)
        self.assertEqual(type(cat), int)

if __name__ == '__main__':
    unittest.main()
