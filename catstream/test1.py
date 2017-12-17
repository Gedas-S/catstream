"""test module for my catstream"""
# pylint: disable=missing-docstring
import unittest
import model
import utils
import torch

class TestStandalone(unittest.TestCase):
    def test_cuda(self):
        self.assertGreater(torch.cuda.device_count(), 0)

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

if __name__ == '__main__':
    unittest.main()
