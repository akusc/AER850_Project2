# -*- coding: utf-8 -*-
"""
Spyder Editor

AER850 - Project 2: CNN Model Development through Training and Validation
By Akus Chhabra

The purpose of this project is to explore Deep Convolution Networks using Image
Recognition to identify crack sizes from images. This script will focus on
developing the CNN model by training and validating the model using the 
supplied crack data.

"""

from tensorflow.keras import layers, models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt

"""
1) Building 2 versions of DCNN model (Diffusion-convolutional neural networks)
2) Classify cracks into 4 classes: Small, Medium, Large, None
"""

#%% Data Processing

## Define size of image
channel = 3
img_height = 100
img_width = 100

## Define path to access Training, Validation, and Testing folders
path = 'C:/Users/akusc/OneDrive/Documents/University_Files/AER850/Project2_Data'
path_train = path + '/Train'
path_valid = path + '/Validation'
path_test = path + '/Test'

## Create an ImageDataGenerator with specified augementation settings
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,       
    shear_range=0.2,          
    zoom_range=0.2,
)
datagen_valid = ImageDataGenerator(rescale=1./255)

## Training Generator
train_data_gen = datagen_train.flow_from_directory(
    path_train,
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=32,
)

## Validation Generator
valid_data_gen = datagen_valid.flow_from_directory(
    path_valid,
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=32,
)

#%% Neural Network Architecture Design & Network Hyperparameters Selection

## Model Definition
model = models.Sequential()

## Define Hidden Layers of Model
model.add(layers.Conv2D(32, (3,3), (1,1), activation='relu', input_shape=(img_height, img_width, 3), padding = 'same'))
model.add(layers.Dropout(.2))
model.add(layers.Conv2D(32, (3,3), (1,1), activation='relu', padding = 'same'))
model.add(layers.Dropout(.2))
model.add(layers.Conv2D(64, (3,3), (1,1), activation='relu', padding = 'same'))
model.add(layers.Dropout(.2))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=1, padding='same'))

model.add(layers.Conv2D(64, (3,3), (1,1), activation='relu', padding = 'same'))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=1, padding='same'))

## Flatten layers and add Dense Layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(.5))
model.add(layers.Dense(4, activation='softmax'))

## Compile model
optimizer = optimizers.Adam()
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_crossentropy', 'accuracy'])
model.summary()

## Fit data using generators and store history
history = model.fit(train_data_gen, validation_data=valid_data_gen, epochs=50) # epochs=50
   
#%% Accuracy and Loss Evaluation of the Model

## Accuracy graph
plt.figure(1)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylim((0,1))
plt.legend(loc='lower right')

## Loss graph
plt.figure(2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation_Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')


# Save trained CNN model 
model.save('my_model6.h5')
 