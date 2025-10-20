from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, Conv2D, Flatten, Dropout, BatchNormalization
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

x_train = (x_train - 127.5)/127.5
x_test = (x_test - 127.5)/127.5
# x reshape -> (60000, 28, 28, 1)
# x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])

# x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2])
# print(x_train.shape, x_test.shape)  #(60000, 28, 28, 1) (10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
# print(y_train.shape, y_test.shape)  #(60000, 10) (10000, 10)

#2. 모델구성
model = Sequential()
# model.add(Conv2D(64, (3,3), strides=1, input_shape=(28,28,1)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (3,3)))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3,3),activation='relu'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(LSTM(128, input_shape=(28,28)))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# # 함수형
# in1   = Input(shape=(28,28,1))
# co1   = Conv2D(64, (3,3), strides=1)(in1)
# bat1  = BatchNormalization()(co1)
# drop1 = Dropout(0.2)(bat1)
# co2   = Conv2D(128, (3,3))(drop1)
# bat2  = BatchNormalization()(co2)
# drop2 = Dropout(0.2)(bat2)
# co3   = Conv2D(32, (3,3), activation='relu')(drop2)
# flat  = Flatten()(co3)
# den1  = Dense(32, activation='relu')(flat)
# bat3  = BatchNormalization()(den1)
# drop3 = Dropout(0.2)(bat3)
# den2  = Dense(16, activation='relu')(drop3)
# out1  = Dense(10, activation='softmax')(den2)
# model = Model(inputs=in1, outputs='out1')

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss', mode='min',
    patience=50, restore_best_weights=True, verbose=1
)

import datetime

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
path1 = './_save/keras36_cnn5/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k36_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='min',
    save_best_only=True, filepath=filepath,
    verbose=1
)

s_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=10000, batch_size=64,
    verbose=1, validation_split=0.2,
    callbacks=[es, mcp]
)
e_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
results = model.predict(x_test)
y_test_arg = np.argmax(y_test.values, axis=1)
results_arg = np.argmax(results, axis=1)
acc = accuracy_score(y_test_arg, results_arg)

print('loss:', np.round(loss[0], 4))
print('ACC :', np.round(acc, 4))
print('time:', np.round(e_time - s_time, 1), 'sec')

# GPU
# loss: 0.0933
# ACC : 0.9791
# time: 261.2 sec

# CPU
# loss: 2.3011
# ACC : 0.1135
# time: 508.9 sec

# GPU(layer 추가)
# loss: 0.047
# ACC : 0.9888
# time: 574.8 sec