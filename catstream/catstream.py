"""do the flask thing where we get cats"""
# note, pytorch is currently not officially supported on windows, but a kind
# person has made it work
# https://github.com/peterjc123/pytorch-scripts
# get it via
# conda install -c peterjc123 pytorch cuda90
# (assuming your GPU supports cuda90)
#
import warnings
from base64 import standard_b64encode
from io import BytesIO
from flask import Flask, render_template, request
from PIL import Image
from model_resnet import (get_network, is_cat, predict_category,
                          CLASSES, PREPROCESSING_TRANSFORM, SIZE_TRANSFORM)

app = Flask(__name__) # pylint: disable=invalid-name

ALLOWED_EXTENSIONS = 'jpg jpe jpeg png gif svg bmp'.split()
MAX_SIZE_MB = 20

warnings.simplefilter('error', Image.DecompressionBombWarning)
app.config['MAX_CONTENT_LENGTH'] = MAX_SIZE_MB * 1024 * 1024

@app.route('/')
def ask_for_cat():
    """homepage: request cat"""
    return render_template('home.html')

@app.route('/cat', methods=['POST'])
def receive_cat():
    """take a post request with an image file, show the uploaded image and say if it's a cat"""
    image = request.files['image']

    # check if a file was received
    if not image:
        return render_template('error.html', message="I lost the cat, did you send?")
    # check if received file is an image
    if image.filename.split('.')[-1] not in ALLOWED_EXTENSIONS:
        return render_template('error.html', message=(
            "Sorry, I meant picture of cat, not actual cat. Does not look like a picture to me. "
            + "I like %s and %s." % (', '.join(ALLOWED_EXTENSIONS[:-1]), ALLOWED_EXTENSIONS[-1])))

    # try to open the image
    try:
        img = SIZE_TRANSFORM(Image.open(image).convert(mode='RGB'))
    except OSError:
        return render_template('error.html', message=
                               "I does not understands your cat. Is your cat corrupted?")
    except Image.DecompressionBombWarning:
        return render_template('error.html', message=
                               "Your cat is very big, perhaps you have a smaller one?")
    # try to run the image through the neural net
    try:
        category_num = predict_category(PREPROCESSING_TRANSFORM(img), cat_net)
    # I thought this was due to running out of bounds, but with resize, it should not
    # I cannot reproduce this after fixing the alpha channel issue, but leaving it in just in case
    except RuntimeError:
        return render_template('error.html', message=
                               "I cannot figure out what went wrong, but you can try another cat!")

    # create message depending on whether we think it's a cat
    category_name = CLASSES[category_num]
    if category_name.find(',') >= 0:
        category_name = ' or'.join(category_name.rsplit(',', 1))
    if is_cat(category_num):
        response_string = "Cat!!! (%s)" % category_name
    else:
        response_string = "I don't think it's a cat, looks like %s to me..." % category_name

    # reencode the resized image before sending back
    byte_stream = BytesIO()
    img.save(byte_stream, 'png', optimize=True)
    byte_stream.seek(0)

    context = {
        'response_string': response_string,
        'image_type': image.headers['Content-Type'],
        'image_data': str(standard_b64encode(byte_stream.read()))[2:-1]
        }
    return render_template('cat.html', **context)

@app.errorhandler(413)
def request_entity_too_large(error): # pylint: disable=unused-argument
    """return the size limit when received file is too large"""
    return render_template('error.html', message=
                           "That is a really big cat. I'm only allowed to handle cats up to %i MB."
                           % MAX_SIZE_MB)

print("Loading CNN, please stand by.")
cat_net = get_network() # pylint: disable=invalid-name
print("CNN loaded, proceeding.")

if __name__ == '__main__':
    app.run(host='0.0.0.0')
