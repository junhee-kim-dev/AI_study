from keras.models import Sequential
from keras.layers import LSTM,Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar100
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

datagen = ImageDataGenerator(
    rescale=1/255., horizontal_flip=True, width_shift_range=0.1, rotation_range=20,
    zoom_range=1.1, fill_mode='nearest'
)
# print(x_train.shape)    #(50000, 32, 32, 3)
# print(x_test.shape)     #(10000, 32, 32, 3)
# print(y_train.shape)    #(50000, 1)
# print(y_test.shape)     #(10000, 1)
aug_x = x_train.copy()
aug_y = y_train.copy()

x_train = x_train/255.
x_test = x_test/255.

augment = datagen.flow(
    aug_x, aug_y, batch_size=100, shuffle=False
)

all_aug_x=[]
all_aug_y=[]
for i in range(500):
    x_bat, y_bat = augment[i]
    all_aug_x.append(x_bat)
    all_aug_y.append(y_bat)
    
aug_x = np.concatenate(all_aug_x, axis=0)
aug_y = np.concatenate(all_aug_y, axis=0)


x_train = np.concatenate((x_train, aug_x))
y_train = np.concatenate((y_train, aug_y))

print(x_train.shape)    #(100000, 32, 32, 3)
print(x_test.shape)     #(10000, 32, 32, 3)
print(y_train.shape)    #(100000, 1)
print(y_test.shape)     #(10000, 1)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)
x_train = x_train.reshape(-1, 32, 32*3)
x_test = x_test.reshape(-1, 32, 32*3)

print(y_train.shape)
# exit()

model = Sequential()
# model.add(Conv2D(64, (3,3), input_shape=(32,32,3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (2,2), activation='relu'))
# model.add(Flatten())
model.add(LSTM(256, input_shape=(32, 32*3)))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

model.compile(
    loss='categorical_crossentropy', optimizer='adam',metrics='acc'
)

es = EarlyStopping(
    monitor='val_acc', mode='auto', verbose=1,
    patience=50, restore_best_weights=True,
)

path = './_save/keras50/cifar100/'
name = '({epoch:04d}-{val_acc:.4f}).hdf5'
file = ''.join([path, name])

mcp = ModelCheckpoint(
    monitor='val_acc', mode='auto', verbose=1,
    save_best_only=True, filepath = file
)

s_time = time.time()
model.fit(
    x_train, y_train, epochs=1000000, batch_size=128,
    verbose=1, validation_split=0.1, callbacks=[es, mcp]
)
e_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
y_test_arg = np.argmax(y_test, axis=1)
results_arg = np.argmax(results, axis=1)
acc = accuracy_score(y_test_arg, results_arg)

print('cifar100')
print('loss :', np.round(loss[0],4))
print('acc  :', np.round(acc, 4))
print('time :', np.round(e_time - s_time, 4))


# loss : 2.5535
# acc  : 0.3684
# time : 254.079



