from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Input
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

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

# print(y_train.shape)    #(50000, 10)
# print(y_test.shape)     #(10000, 10)

x_train = x_train/255.
x_test = x_test/255.
x_train = x_train.reshape(-1,32,32*3)
x_test = x_test.reshape(-1,32,32*3)

model = Sequential()
# model.add(Conv2D(64, (2,2), input_shape=(32,32,3), activation='relu', padding='same'))
# model.add(MaxPooling2D())
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Conv2D(128, (2,2), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Conv2D(64, (3,3), activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Flatten())
model.add(LSTM(128, input_shape=(32, 32*3)))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# input = Input(shape=(32,32,3))
# conv1 = Conv2D(64, 2, activation='relu', padding='same')(input)
# maxp1 = MaxPooling2D()(conv1)
# batc1 = BatchNormalization()(maxp1)
# drop1 = Dropout(0.2)(batc1)
# conv2 = Conv2D(128, 2, activation='relu', padding='same')(drop1)
# batc2 = BatchNormalization()(conv2)
# drop2 = Dropout(0.2)(batc2)
# conv3 = Conv2D(64, 3, activation='relu', padding='same')(drop2)
# maxp2 = MaxPooling2D()(conv3)
# batc3 = BatchNormalization()(maxp2)
# drop3 = Dropout(0.2)(batc3)
# conv4 = Conv2D(64, 3, activation='relu', padding='same')(drop3)
# batc4 = BatchNormalization()(conv4)
# drop4 = Dropout(0.2)(batc4)
# conv5 = Conv2D(32, 3, activation='relu', padding='same')(drop4)
# batc5 = BatchNormalization()(conv5)
# drop5 = Dropout(0.2)(batc5)
# flatt = Flatten()(drop5)
# outpu = Dense(10, activation='relu')(flatt)
# model = Model(inputs=input, outputs=outpu)

model.compile(
    loss='categorical_crossentropy', optimizer='adam',metrics='acc'
)

es = EarlyStopping(
    monitor='val_acc', mode='auto', verbose=1,
    restore_best_weights=True, patience=50
)

path = './_save/keras40/cifar10/'
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

print('cifar10')
print('loss :', np.round(loss[0], 4))
print('acc  :', np.round(acc,4))
print('time :', np.round(e_time - s_time, 1), 'sec')
