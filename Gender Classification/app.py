from flask import *

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

MODEL_PATH ='Gender-Model.h5'
model = load_model(MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(250,250))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)

    if preds[0][0] == 1:
         classification = "He is a Male 🏃"
    else:
         classification = "She is a Female 💃"

    return render_template('index.html', prediction = classification)

if __name__ == '__main__':
    app.run(debug=True, port=3000)