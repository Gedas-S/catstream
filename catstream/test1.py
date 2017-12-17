"""test module for my catstream"""
# pylint: disable=missing-docstring
import unittest
import model
import utils
import torch

class Test1(unittest.TestCase):
    def test_cuda(self):
        self.assertGreater(torch.cuda.device_count(), 0)

    def test_model(self):
        net = model.train_network()
        acc = utils.calculate_accuracy(utils.cifar10_testdata(), net)
        utils.print_accuracy(acc)
        self.assertGreater(acc, 0.5)

if __name__ == '__main__':
    unittest.main()
