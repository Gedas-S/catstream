"""test module for my catstream"""
# pylint: disable=missing-docstring
import unittest
from io import BytesIO
import torch
from PIL import Image
from werkzeug.datastructures import FileStorage
import model_resnet as model
import catstream

DOT64 = b'iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAJklEQVR42nXIsQkAIADAsNb/f66LIoJmjBVobKXBiWXw8MnymnICEGsMAo+vrWIAAAAASUVORK5CYII='


class TestStandalone(unittest.TestCase):

    def test_cuda(self):
        self.assertGreater(torch.cuda.device_count(), 0)

    def test_b64_image_encoding(self):
        from base64 import standard_b64encode
        with open('test_files/dot.png', 'rb') as file: # taken from wiki, public domain
            rdata = FileStorage(file)
            img = Image.open(rdata)
            stream = BytesIO()
            img.save(stream, 'png', optimize=True)
            stream.seek(0)
            bimg = standard_b64encode(stream.read())
            self.assertEqual(bimg, DOT64)


class TestResnetNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.net = catstream.cat_net

    def test_resnet_image(self):
        # I just drew the image and hereby release it into the public domain :P
        img = Image.open('test_files/test_image.jpg')
        img = model.PREPROCESSING_TRANSFORM(model.SIZE_TRANSFORM(img))
        cat = model.predict_category(img, self.net)
        self.assertEqual(type(cat), int)


class TestFlask(unittest.TestCase):

    def setUp(self):
        catstream.app.testing = True
        self.app = catstream.app.test_client()

    def test_root_request(self):
        response = self.app.get('/')
        self.assertGreater(str(response.data).find('Can I has cat?'), 0)

    def image_test(self, file_name, message):
        with open('test_files/'+file_name, 'rb') as file:
            file_data = FileStorage(file)
            response = self.app.post('/cat',
                                     content_type='multipart/form-data',
                                     data={'image': file_data},
                                     follow_redirects=True)
            response_string = str(response.data)
            try:
                self.assertGreater(response_string.find(message), 0)
            except AssertionError:
                print(response_string)
                raise

    def test_blank_image(self):
        self.image_test('bad.jpg', 'Is your cat corrupted')

    def test_test_image(self):
        self.image_test('test_image.jpg', "I don&#39;t think it&#39;s a cat")

    def test_not_image(self):
        self.image_test('txt.txt', 'Does not look like a picture to me. I like jpg')


if __name__ == '__main__':
    unittest.main()
