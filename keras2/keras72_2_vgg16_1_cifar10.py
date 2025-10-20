import numpy as np

from keras.models import Sequential
from keras.layers import Dense, AveragePooling2D
import tensorflow as tf
import random

SEED=333
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)    #(50000, 32, 32, 3)
# print(x_test.shape)     #(10000, 32, 32, 3)
# print(y_train.shape)    #(50000, 1)
# print(y_test.shape)     #(10000, 1)

# ohe = OneHotEncoder(sparse=False)
# y_train = ohe.fit_transform(y_train)
# y_test = ohe.transform(y_test)

# print(y_train.shape)    #(50000, 100)
# print(y_test.shape)     #(10000, 100)

x_train = x_train/255.
x_test = x_test/255.

from keras.applications import VGG16

vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3)
)

##########
# False와 AveragePooling2D 비교
# vgg16.trainable=False   # 가중치 동결
vgg16.trainable=True   # 가중치 동결

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(AveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.summary()

##########
# Flatten 과 AveragePooling2D 비교

model.copile(loss='sparse_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=100, verbose=2, )

loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print(f"[Final] Loss {loss:.4f} | Acc {acc:.4f}")