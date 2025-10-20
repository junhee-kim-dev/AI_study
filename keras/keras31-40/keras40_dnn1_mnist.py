# CNN -> DNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import time

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

# print(x_train.shape[0]) #60000
# print(x_train.shape[1]) #28
# print(x_train.shape[2]) #28

x_train = x_train/255.
x_test = x_test/255.
# print(np.max(x_train), np.min(x_train)) #1.0 0.0
# print(np.max(y_train), np.min(y_train)) #9 0

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
y_train = y_train.reshape(60000, 1)
y_test = y_test.reshape(10000, 1)

# print(x_train.shape[0]) #60000
# print(x_train.shape[1]) #784
# print(x_train.shape[2])

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

model = Sequential()
model.add(Dense(128, input_dim=784, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2, seed=42))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2, seed=42))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2, seed=42))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2, seed=42))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

es = EarlyStopping(
    monitor='val_loss', mode='auto',
    patience=100, verbose=1,
    restore_best_weights=True
)
import datetime
path = './_save/keras40/digits/CPU_'
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filename = '({epoch:04d}_{val_loss:.4f}).hdf5'
filepath = ''.join([path, 'digits_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto',
    save_best_only=True, verbose=1,
    filepath=filepath
)

s_time = time.time()
hist = model.fit(x_train, y_train, epochs=10000000, 
          batch_size=256, verbose=2,
          validation_split=0.2, callbacks=[es, mcp])
e_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
result_arg = np.argmax(results, axis=1)
y_test_arg = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_arg, result_arg)

print('Acc :', np.round(acc, 4))
print('time:', np.round(e_time - s_time, 1), 'sec')

# DNN_GPU
# Acc : 0.9831
# time: 215.7 sec

# DNN_CPU
# Acc : 0.9822
# time: 70.9 sec