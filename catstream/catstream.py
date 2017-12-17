"""do the flask thing where we get cats"""
from base64 import standard_b64encode
from flask import Flask, render_template, request
from PIL import Image
from model import get_network
from utils import cat, preprocess_image, CIFAR10_CLASSES

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
        img = Image.open(image)
        is_cat = cat(preprocess_image(img), cat_net)
    except OSError:
        return "I does not understands your cat :( Is your cat corrupted?"
    except RuntimeError:
        return "I cannot make out anything, could you send a bigger photo?"

    if CIFAR10_CLASSES[is_cat] == 'cat':
        response_string = "Cat!!!"
    else:
        response_string = ("I don't think it's a cat, looks like a %s to me..."
                           % CIFAR10_CLASSES[is_cat])

    image.seek(0)
    context = {
        'response_string': response_string,
        'image_type': image.headers['Content-Type'],
        'image_data': str(standard_b64encode(image.read()))[2:-1]
        }
    return render_template('cat.html', **context)


if __name__ == '__main__':
    print("Loading cnn, please stand by.")
    cat_net = get_network() # pylint: disable=invalid-name

    app.run(host='0.0.0.0')
