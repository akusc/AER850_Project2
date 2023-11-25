# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:31:33 2023

AER850 - Project 2: Testing CNN Model
By Akus Chhabra

The purpose of this script is to test the developed CNN model using the 
supplied testing crack data.

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt

#%% Model Testing of New Images

img_height = 100
img_width = 100

## Load stored model from training and validation fitting
model = tf.keras.models.load_model('my_model6.h5')

## Access file, convert image to img_to_array, and normalize
#path = 'C:/Users/akusc/OneDrive/Documents/University_Files/AER850/Project2_Data/Test/Large/Crack__20180419_13_29_14,846.bmp'
path = 'C:/Users/akusc/OneDrive/Documents/University_Files/AER850/Project2_Data/Test/Medium/Crack__20180419_06_19_09,915.bmp'
img = load_img(path, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255

## Model predictions 
predictions = model.predict(img_array)

## Output Class labels
predicted_class = np.argmax(predictions)
predicted_prob = np.max(predictions)

## Dictionary of labels
labels = {0: 'Large', 1: 'Medium', 2: 'None', 3: 'Small'}

## Test image label
#true_label = 'Large'
true_label = 'Medium'

## Map prediction to respective label 
predicted_label = labels[predicted_class]
output = f"True Label: {true_label}\nPredicted Label: {predicted_label}\nProbability: {predicted_prob:.2f}"


## Plot image
plt.imshow(img)
plt.title(output)
plt.axis('off')

## Display probabilities on image
for i, prob in enumerate(predictions[0]):
    label = labels[i]
    text_color = 'green' if label == predicted_label else 'red'
    plt.text(5, 20+20*i, f"{label}: {prob:.2f}", bbox=dict(facecolor='black', alpha=0.5), fontsize=10, color=text_color)
plt.show()
