from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from flask_cors import CORS
import numpy as np
import base64
from io import BytesIO
from PIL import Image 
import cv2


app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
model = tf.keras.models.load_model('MNIST_model')

def scale(image):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image

@app.route('/guess', methods=['POST'])
def default_route():
    size = 28, 28
    base64_img = request.get_json()['img']
    img = Image.open(BytesIO(base64.b64decode(base64_img)))
    img = img.resize(size, Image.ANTIALIAS).convert('LA')
    
    image_array = np.asarray(img)   
    image_array = np.delete(image_array, 0,2)
    image_array = np.squeeze(image_array, axis=2)
    image_array = np.divide(image_array, 255)

    prediction = model.predict(np.asarray([image_array]))
    print((np.argmax(prediction[0])))
    return str(np.argmax(prediction[0]))

@app.route('/', methods=['GET'])
def render_page():
    return render_template("app.html")
    
if __name__ == '__main__':
    app.debug=True
    app.run()