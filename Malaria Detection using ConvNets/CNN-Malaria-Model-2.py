# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:30:31 2021

@author: soory
"""

##-----------------------------------------------------------------------------
# TITLE: CNN MALARIA CELL CLASSIFICATION
##-----------------------------------------------------------------------------

# IMPORTING THE LIBRARIES
##------------------------

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

import numpy as np  # Data manipulation
import pandas as pd # Dataframe manipulation 
import matplotlib.pyplot as plt # Plotting the data and the results
import matplotlib.image as mpimg # For displaying imagees
from keras import models
from keras import layers
import keras.preprocessing  as kp
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import optimizers
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

# DATA PRE-PROCESSING
##--------------------

# PREPROCESSING THE TRAINING SET
train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=30,
                                    shear_range=0.3,
                                    zoom_range=0.3)

training_set = train_datagen.flow_from_directory(r"D:\001_Data\END 2 END DL\01_Datasets\Malaria-Cell-Images\Malaria Cells\training_set",
                                                 target_size = (150, 150),
                                                 batch_size = 48,
                                                 class_mode = 'binary')


# PREPROCESSING THE TEST SET
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(r"D:\001_Data\END 2 END DL\01_Datasets\Malaria-Cell-Images\Malaria Cells\testing_set",
                                            target_size = (150, 150),
                                            batch_size = 48,
                                            class_mode = 'binary')


# BUILDING THE CNN MODEL
##----------------------

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())   


# TRAINING THE CNN MODEL
##------------------------

history=model.fit(training_set,steps_per_epoch=70,epochs=50,
                  validation_data=test_set,validation_steps=50)


# ACCURACY & LOSS PLOTS
##----------------------

# list all data in history
print(history.history.keys())

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(50)

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# TESTING THE MODEL ON CUSTOM IMAGES
## -----------------------------------

from keras.preprocessing import image
test_image = image.load_img(r"D:\001_Data\END 2 END DL\01_Datasets\Malaria-Cell-Images\Malaria Cells\testing_set\Uninfected\C6NThinF_IMG_20150609_122725_cell_185.png",
                            target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

print(training_set.class_indices)
print(result)

if result[0][0] == 1:
    prediction = 'NOT INFECTED'
else:
    prediction = 'INFECTED'
print(prediction)

model.save('Malaria-Model-2.h5')


# THE END
#-----------

