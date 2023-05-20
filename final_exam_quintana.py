import streamlit as st
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive')

#import needed libraries

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

# import library
data_set = '/content/drive/MyDrive/Colab Notebooks/Final exam/Agricultural-crops'

folder = os.listdir(data_set)

#define the size of input images
image_size = (130, 130)
#define the size of training and validation dataset
batch_size = 10
#use keras imagedatagenerator to load the data
data_gen = ImageDataGenerator(rescale=1./255,
    validation_split=0.3)

train_set = data_gen.flow_from_directory(data_set, target_size=image_size,batch_size=batch_size,subset='training')

validation_set = data_gen.flow_from_directory(data_set,target_size=image_size,batch_size=batch_size,subset='validation')

lb = LabelBinarizer()
lb.fit(train_set.classes)

num_classes = train_set.num_classes

#baseline model
model = Sequential()
model.add(Flatten(input_shape=train_set.image_shape))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.summary()

#improve model using known techniques

model = Sequential()
model.add(Flatten(input_shape=train_set.image_shape))
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax'))

#compile model
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#save the best model
bestmodel = '/content/drive/MyDrive/Colab Notebooks/Final exam/best model/bestmodel.h5'
checkpoint = ModelCheckpoint(bestmodel, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#train model
results2 = model.fit(train_set, epochs=100, validation_data=validation_set,callbacks=[checkpoint])

#load save model
model = load_model(bestmodel)

# Load sample image from Agricultural-crops/Coffee-plant and preprocess it
img = Image.open('/content/drive/MyDrive/Colab Notebooks/Final exam/Agricultural-crops/Coffee-plant/images54.jpg').resize((130, 130))
X = np.array(img) / 130.0
X = np.expand_dims(X, axis=0)

# Make a prediction using the save model
predictions = model.predict(X)

# Print the category and its probability
category = ['coffee-plant', 'lemon']
predicted_category = category[np.argmax(predictions)]
probability = np.max(predictions)
print(f'Predicted class: {predicted_category}')
print(f'Probability: {probability}')

# Show the image
plt.imshow(img)
plt.show()

# Load sample image from Agricultural-crops/Lemon and preprocess it
img = Image.open('/content/drive/MyDrive/Colab Notebooks/Final exam/Agricultural-crops/Lemon/image25.jpeg').resize((130, 130))
X = np.array(img) / 130.0
X = np.expand_dims(X, axis=0)

# Make a prediction using the save model
predictions = model.predict(X)

# Print the category and its probability
category = ['coffee-plant', 'lemon']
predicted_category = category[np.argmax(predictions)]
probability = np.max(predictions)
print(f'Predicted class: {predicted_category}')
print(f'Probability: {probability}')

# Show the image
plt.imshow(img)
plt.show()
