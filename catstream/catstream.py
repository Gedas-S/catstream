"""do the flask thing where we get cats"""
from flask import Flask, render_template, request
from PIL import Image
from model import get_network, cat
from utils import preprocess_image, CIFAR10_CLASSES

app = Flask(__name__) # pylint: disable=invalid-name

ALLOWED_EXTENSIONS = 'jpg jpe jpeg png gif svg bmp'.split()

@app.route('/')
def ask_for_cat():
    """homepage: request cat"""
    return render_template('home.html')

@app.route('/cat', methods=['POST'])
def receive_cat():
    """take a post request with an image file, show the uploaded image and say if it's a cat"""
    image = request.files['image']
    if not image:
        return "I lost the cat, did you send?"
    if image.filename.split('.')[-1] not in ALLOWED_EXTENSIONS:
        return ("Sorry, I meant picture of cat, not actual cat. Does not look like a picture to me. I like %s and %s."
                % (', '.join(ALLOWED_EXTENSIONS[:-1]), ALLOWED_EXTENSIONS[-1]))
    try:
        image = Image.open(image)
        is_cat = cat(preprocess_image(image), cat_net)
    except OSError:
        return "I does not understands your cat :( Is your cat corrupted?"
    return CIFAR10_CLASSES[is_cat]


if __name__ == '__main__':
    print("Loading cnn, please stand by.")
    cat_net = get_network() # pylint: disable=invalid-name

    app.run(host='0.0.0.0')
