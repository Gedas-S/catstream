"""do the flask thing where we get cats"""
# note, pytorch is currently not officially supported on windows, but a kind
# person has made it work
# https://github.com/peterjc123/pytorch-scripts
# get it via
# conda install -c peterjc123 pytorch cuda90
# (assuming your GPU supports cuda90)
#
# also note, this is mostly following the pytorch tutorial at
# http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
#
from base64 import standard_b64encode
from io import BytesIO
from flask import Flask, render_template, request
from PIL import Image
from model_resnet import (get_network, is_cat, predict_category,
                          CLASSES, PREPROCESSING_TRANSFORM, SIZE_TRANSFORM)

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

    # check if a file was received
    if not image:
        return render_template('error.html', message="I lost the cat, did you send?")
    # check if received file is an image
    if image.filename.split('.')[-1] not in ALLOWED_EXTENSIONS:
        return render_template('error.html', message=(
            "Sorry, I meant picture of cat, not actual cat. Does not look like a picture to me. I like %s and %s."
            % (', '.join(ALLOWED_EXTENSIONS[:-1]), ALLOWED_EXTENSIONS[-1])))

    # try to open the image
    try:
        img = Image.open(image)
        img = SIZE_TRANSFORM(img)
    except OSError:
        return render_template('error.html', message=
                               "I does not understands your cat. Is your cat corrupted?")
    # try to run the image through the neural net
    try:
        category_num = predict_category(PREPROCESSING_TRANSFORM(img), cat_net)
    except RuntimeError: # this usually happens when the convolution layers run out of bounds
        return render_template('error.html', message=
                               "I cannot make out anything, could you send a bigger photo?")

    # create message depending on whether we think it's a cat
    if is_cat(category_num):
        response_string = "Cat!!! (%s)" % CLASSES[category_num]
    else:
        response_string = ("I don't think it's a cat, looks like a %s to me..."
                           % CLASSES[category_num])

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

print("Loading CNN, please stand by.")
cat_net = get_network() # pylint: disable=invalid-name
print("CNN loaded, proceeding.")

if __name__ == '__main__':
    app.run(host='0.0.0.0')
