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

# Loop through the folder and add up the scores
folder_path = "data/non_face"
total_score = 0
total_images = 0
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        img_path = os.path.join(folder_path, filename)
        img = keras.utils.load_img(img_path, target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis
        predictions = model.predict(img_array)
        score = float(predictions[0][0])
        #print(f"This image is {100 * (1 - score):.2f}% portrait and {100 * score:.2f}% non-portrait.")
        print(score)
        total_score += score
        total_images += 1

# Calculate the average score
if total_images > 0:
    average_score = total_score / total_images
    print(f"Average score: {average_score:.2f}")
else:
    print("No images found in the folder.")