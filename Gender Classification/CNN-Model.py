##-----------------------------------------------------------------------------
# TITLE: CNN DOGS AND CAT BINARY CLASSIFIER
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

# DATA PRE-PROCESSING
##--------------------

# PREPROCESSING THE TRAINING SET
train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=30,
                                    shear_range=0.3,
                                    zoom_range=0.3)

training_set = train_datagen.flow_from_directory(r"D:\001_Data\END 2 END DL\01_Datasets\Malaria-Cell-Images\train",
                                                 target_size = (135, 135),
                                                 batch_size = 48,
                                                 class_mode = 'binary')


# PREPROCESSING THE TEST SET
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(r"D:\001_Data\END 2 END DL\01_Datasets\Malaria-Cell-Images\test",
                                            target_size = (135, 135),
                                            batch_size = 48,
                                            class_mode = 'binary')


# BUILDING THE CNN MODEL
##----------------------

kernel_s=(3,3) # The size of kernel

# ConvNet
model=models.Sequential()
model.add(layers.Conv2D(32,kernel_s,activation='relu',input_shape=(135,135,3),
                        kernel_regularizer=regularizers.l2(0.001),padding="same"))
model.add(layers.MaxPooling2D((2,2),strides=2))

model.add(layers.(Conv2D(filters=64, kernel_size=(3,3),input_shape=(134,131,3),
                         activation='relu',padding="same"))
model.add(layers.MaxPooling2D((2,2),strides=2))

model.add(layers.Conv2D(Conv2D(filters=128, kernel_size=(3,3),input_shape=(134,131,3),
                               activation='relu',padding="same"))
model.add(layers.MaxPooling2D((2,2),strides=2))

model.add(layers.Conv2D(Conv2D(filters=256, kernel_size=(3,3),input_shape=(134,131,3),
                               activation='relu',padding="same"))
model.add(layers.MaxPooling2D((2,2),strides=2))

model.add(layers.Conv2D(Conv2D(filters=512, kernel_size=(3,3),input_shape=(134,131,3),
                               activation='relu',padding="same"))
model.add(layers.MaxPooling2D((2,2),strides=2))

model.add(layers.Flatten())

# DENSE LAYERS
model.add(Dense(128,activation='relu'))

# DROPOUT LAYER
model.add(Dropout(0.2))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.2))

# OUTPUT LAYER
model.add(Dense(1,activation='sigmoid'))

model.summary()


# TRAINING THE CNN MODEL
##------------------------

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(training_set,steps_per_epoch=70,epochs=30,
                  validation_data=test_set,validation_steps=50)


# ACCURACY & LOSS PLOTS
##----------------------

# list all data in history
print(history.history.keys())

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(30)

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
plt.save("Gender-Accuracy.png")

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.save("Gender-Loss.png")

# TESTING THE MODEL ON CUSTOM IMAGES
## -----------------------------------

from keras.preprocessing import image
test_image = image.load_img(r"D:\001_Data\END 2 END DL\01_Datasets\Gender\Validation\female\113421.jpg.jpg",
                            target_size = (250, 250))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
prob = model.predict_proba(test_image)

print(training_set.class_indices)
print(result)

if result[0][0] == 1:
    prediction = 'The prediction is a MALE'
else:
    prediction = 'The prediction is a FEMALE'
print(prediction)

model.save('Gender-Model.h5')


# THE END
-----------
