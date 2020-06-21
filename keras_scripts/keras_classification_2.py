# -*- coding: utf-8 -*-

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#%%
from os import listdir
from os.path import isfile, join
import cv2

X = []
y = []


for path, label in [('./data/clock/', 1), ('./data/done/', 0)]: #('./data/anti/', -1), 
    for f in listdir(path):
        img = cv2.imread(join(path, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[120:380, 150:450]
        img = np.expand_dims(img, axis=2)
        X.append(img)
        y.append(label)
X = np.array(X)
y = np.array(y)

num_classes = 2

#%%
    
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#%%

import matplotlib.pyplot as plt

plt.imshow(X_train[0,:,:,0], cmap='gray')


input_shape = (260, 300, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=16,
          epochs=10,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose = 0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
