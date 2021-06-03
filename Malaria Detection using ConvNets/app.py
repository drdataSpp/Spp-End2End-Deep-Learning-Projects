from flask import Flask, request, render_template

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

MODEL_PATH ='Malaria-Model-2.h5'
model = load_model(MODEL_PATH)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)

    if preds[0][0] == 1:
         classification = "You are not infected by Malaria 😇"
    else:
         classification = "You are infected by Malaria. Please consult a doctor."

    return render_template('index.html', prediction = classification)

if __name__ == '__main__':
    app.run(debug=True, port=3000)