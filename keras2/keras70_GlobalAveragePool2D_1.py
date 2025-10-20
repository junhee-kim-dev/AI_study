from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import time

from keras.datasets import mnist
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# x reshape -> (60000, 28, 28, 1)
# x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)

# x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
# print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
# print(y_train.shape, y_test.shape)  #(60000, 10) (10000, 10)

#2. 모델구성
model = Sequential()
model.add(Conv2D(100, (2,2), strides=1, input_shape=(10,10,1)))
model.add(Conv2D(50, (2,2)))
model.add(Conv2D(30, (2,2), padding='same'))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 26, 26, 64)        640       
#  conv2d_1 (Conv2D)           (None, 24, 24, 32)        18464     
#  conv2d_2 (Conv2D)           (None, 22, 22, 16)        4624      
#  flatten (Flatten)           (None, 7744)              0
#  dense (Dense)               (None, 16)                123920    
#  dense_1 (Dense)             (None, 16)                272
#  dense_2 (Dense)             (None, 10)                170
# =================================================================
# Total params: 148,090
# Trainable params: 148,090
# Non-trainable params: 0
# _________________________________________________________________

