from flask import Flask, request, render_template

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

MODEL_PATH ='Mushroom_Model.h5'
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)

    preds=np.argmax(model.predict(image))

    if preds==0:
        classification = "PREDICTION: Edible"
    else:
        classification = "PREDICTION: Poisonous"

    return render_template('index.html', prediction = classification)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
