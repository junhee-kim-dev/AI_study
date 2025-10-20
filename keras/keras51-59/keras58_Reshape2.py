#36_6 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, Reshape, LSTM
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import time

from tensorflow.keras.datasets import mnist
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

x_train = x_train/255.              # 255로 나눈다. 맨 뒤에 '.'은 부동소수점 연산 때문에 쓴것
x_test = x_test/255.

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#2. 모델구성
model = Sequential()
model.add(LSTM(60, input_shape=(28,28), return_sequences=True))
model.add(Reshape(target_shape=(28,6,10)))
model.add(Conv2D(64, (3,3), input_shape=(28,6,10), padding='same'))
model.add(Conv2D(124, (3,3)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(32, (3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
