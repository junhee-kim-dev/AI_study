from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)
datagen = ImageDataGenerator(
    rescale=1/255.,
    horizontal_flip=True, width_shift_range=0.1,
    zoom_range=1.2, rotation_range=20,
    fill_mode='nearest'
)
augment_size=40000
rad_idx = np.random.randint(60000, size=40000)
aug_x = x_train[rad_idx].copy()
aug_y = y_train[rad_idx].copy()

aug_x = aug_x.reshape(-1,28,28,1)

aug_x = datagen.flow(
    aug_x, aug_y, batch_size=augment_size, shuffle=False
).next()[0]

x_train = x_train/255.
x_test = x_test/255.
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# print(x_train.shape, x_test.shape) #(60000, 28, 28, 1) (10000, 28, 28, 1)
# print(aug_x.shape, aug_y.shape)#(40000, 28, 28, 1) (40000,)
# print(y_train.shape, y_test.shape)#(60000,) (10000,)

x_train = np.concatenate((x_train, aug_x))
y_train = np.concatenate((y_train, aug_y))

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
# (100000, 28, 28, 1) (10000, 28, 28, 1)
# (100000, 10) (10000, 10)


model = Sequential()
model.add(Conv2D(64, (3,3), strides=1, input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3)))
model.add(Conv2D(32, (3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

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
path1 = './_save/keras50/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'k50_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='min',
    save_best_only=True, filepath=filepath,
    verbose=1
)

s_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=100000, batch_size=128,
    verbose=2, validation_split=0.2,
    callbacks=[es, mcp]
)
e_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
results = model.predict(x_test)
y_test_arg = np.argmax(y_test, axis=1)
results_arg = np.argmax(results, axis=1)
acc = accuracy_score(y_test_arg, results_arg)

print('loss:', np.round(loss[0], 4))
print('ACC :', np.round(acc, 4))
print('time:', np.round(e_time - s_time, 1), 'sec')

# loss: 0.0466
# ACC : 0.9863
# time: 165.7 sec