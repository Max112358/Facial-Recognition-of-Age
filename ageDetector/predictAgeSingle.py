import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
import tensorflow_hub as hub
from file_utils import find_highest_numbered_file
from keras.models import load_model

image_size = (300, 300)
larger_input = image_size + (3,)
batch_size = 128


def number_to_string(num):
    if num == 7:
        return "60+"
    elif num == 6:
        return "48-53"
    elif num == 5:
        return "38-43"
    elif num == 4:
        return "25-32"
    elif num == 3:
        return "15-20"
    elif num == 2:
        return "8-13"
    elif num == 1:
        return "4-6"
    else:
        return "0-2"



if os.path.exists("model5Categorical.keras"):
    # Load the weights if the file exists
    model = load_model("model5Categorical.keras")
    print("Previous weights found, using those")
else:
    print("no model found")


model.summary()


img = keras.utils.load_img("catagorized/50/87018_1964-10-27_2014.jpg", target_size=image_size)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)

'''
#regressive predictions
score = float(predictions[0][0])
print(predictions[0])
print(score)
print("I think this person is age: " + str(score))
'''

#catagorical predictions
classes = np.argmax(predictions, axis = 1)
#print(predictions)
print("I think this image belongs to: " + number_to_string(classes[0]) + " age range")

