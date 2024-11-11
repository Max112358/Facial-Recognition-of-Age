import os
import tensorflow as tf
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

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

# Loop through the folder and count the number of dogs and cats
folder_path = "data/face"
face = 0
nonface = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        img_path = os.path.join(folder_path, filename)
        img = keras.utils.load_img(img_path, target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
        predictions = model.predict(img_array)
        score = float(keras.ops.sigmoid(predictions[0][0]))
        if score >= 0.6:
            nonface += 1
        else:
            face += 1

# Print the ratio
total_images = face + nonface
face_ratio = face / total_images
nonface_ratio = nonface / total_images
print(f"Ratio of face to nonface: {face_ratio:.2f} : {nonface_ratio:.2f}")