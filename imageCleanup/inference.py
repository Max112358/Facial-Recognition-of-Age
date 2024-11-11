import os
import tensorflow as tf
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
import imageSorterModel

image_size = (180, 180)
batch_size = 128

def create_model(input_shape, num_classes):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))  # Change output activation to sigmoid for binary classification    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model(input_shape=image_size + (3,), num_classes=2)
model.load_weights("save_at_5.keras")


img = keras.utils.load_img("data/non_face/2034705_1915-10-30_1942.jpg", target_size=image_size)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
#score = float(keras.ops.sigmoid(predictions[0][0]))
#print(f"This image is {100 * (1 - score):.2f}% non-portrait and {100 * score:.2f}% portrait.")

#you dont want to sigmoid a already sigmoided output, so use this one
score = float(predictions[0][0])
print(f"This image is {100 * (1 - score):.2f}% portrait and {100 * score:.2f}% non-portrait.")