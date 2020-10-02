from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from flask_cors import CORS
import numpy as np
import base64
from io import BytesIO
from PIL import Image 


app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
model = tf.keras.models.load_model('MNIST_model')

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
    rounded_predictions = np.round_(prediction[0], decimals=2)

    return jsonify({
        'guess': float(np.argmax(prediction[0])),
        'prediction': rounded_predictions.tolist()
        })

@app.route('/', methods=['GET'])
def render_page():
    return render_template("app.html")

@app.after_request
def add_header(response):
    response.cache_control.max_age = 0
    return response
    
if __name__ == '__main__':
    app.run(debug=True)