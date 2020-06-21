# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras

#%%
from os import listdir
from os.path import isfile, join
import cv2

X = []
y = []


for path, label in [('./data/correct/', 1), ('./data/incorrect_1/', 0)]: #('./data/anti/', -1), 
    for f in listdir(path):
        img = cv2.imread(join(path, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[150:600,250:650]/255
        img = np.expand_dims(img, axis=2)
        X.append(img)
        y.append(label)
X = np.array(X)
y = np.array(y)

# num_classes = 2 # for softmax
# y = keras.utils.to_categorical(y, num_classes)
#%%
    
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)

#%%

import matplotlib.pyplot as plt

plt.imshow(img[:, :, 0], cmap='gray')

#%%

# Model 1:

inputs = keras.Input(shape=(450, 400, 1))
# Apply some convolution and pooling layers
x = keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(inputs)
x = keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
x = keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(x)
x = keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
x = keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(x)

# Apply global average pooling to get flat feature vectors
x = keras.layers.GlobalAveragePooling2D()(x)

# Add a dense classifier on top
num_classes = 1
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

#%%

# Model 2:

# inputs = keras.Input(shape=(450, 400, 1))
# # Apply some convolution and pooling layers
# x = keras.layers.Conv2D(filters=32, kernel_size=(10, 10), activation="relu")(inputs)
# x = keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
# x = keras.layers.Conv2D(filters=32, kernel_size=(10, 10), activation="relu")(x)
# # x = keras.layers.MaxPooling2D(pool_size=(3, 3))(x)
# x = keras.layers.GlobalAveragePooling2D()(x)


# # Add a dense classifier on top
# num_classes = 2
# outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

# model = keras.Model(inputs=inputs, outputs=outputs)

# print(model.summary())

#%%

# Model 3:

# num_classes = 2

# model = keras.Sequential(
#     [
#         keras.Input(shape=(450, 400, 1)),
#         keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         keras.layers.Flatten(),
#         keras.layers.Dropout(0.5),
#         keras.layers.Dense(num_classes, activation="softmax"),
#     ]
# )

# model.summary()

#%%
model.compile(optimizer='adam', loss='mse', metrics=["accuracy", tf.keras.metrics.AUC()]) #categorical_crossentropy

model.fit(X_train, y_train, batch_size=1, epochs=10)#validation_split=0.1