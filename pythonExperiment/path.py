# Get the directory where NumPy is installed
libraryPath = 'C:/Users/maxjo/AppData/Local/packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages' 

# Add the NumPy directory to the beginning of sys.path
#sys.path.insert(0, libraryPath)


import sys
import os

# Print the current sys.path
print(sys.path)
print('------------------------------------------------------------------------------')

import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt