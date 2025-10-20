from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar100
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

path = '_data/_save_npy/keras50/catdog/'
x = np.load(path + 'x.npy')
y = np.load(path + 'y.npy')
test = np.load(path + 'test.npy')

# print(x.shape)      #(40000, 100, 100, 3)
# print(y.shape)      #(40000,)
# print(test.shape)   #(25000, 100, 100, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)
x_train = x_train.reshape(-1,100,300)
x_test = x_test.reshape(-1,100,300)


print('가보자.')
model = Sequential()
# model.add(Conv2D(32, 2, input_shape=(100,100,3), activation='relu', padding='same'))
# model.add(MaxPooling2D())
# model.add(Conv2D(64, 3, activation='relu'))
# model.add(Conv2D(32, 3, activation='relu'))
# model.add(Flatten())
model.add(LSTM(128, input_shape=(100,300)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(
    monitor='val_acc', mode='max', verbose=1,
    patience=10, restore_best_weights=True,
)
import datetime
path1 = './_data/kaggle/CatDog/mcp/'
date = datetime.datetime.now()
date = date.strftime('%H%M')
name = '({epoch:04d}_{val_loss:.4f}).hdf5'
f_path = ''.join([path1, 'k50_', date, '_', name])

mcp = ModelCheckpoint(
    monitor='val_acc', mode='max', verbose=1,
    save_best_only=True, filepath=f_path
)

s_time = time.time()
model.fit(
    x_train, y_train, epochs=100, batch_size=128,
    verbose=1, validation_split=0.2, callbacks=[es, mcp]
)
e_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
results_round = np.round(results)
acc = accuracy_score(y_test, results_round)

print('CatDog')
print('Loss :', np.round(loss[0],4))
print('Acc  :', np.round(acc,4))
print('time :', np.round(e_time - s_time, 2), 'sec')