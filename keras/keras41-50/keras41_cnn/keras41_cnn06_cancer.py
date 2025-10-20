from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random

#1. 데이터
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

x = dataset.data    #(569, 30)
y = dataset.target  #(569,)


r = 42 #random.randint(1, 10000)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=r,
)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# exit()
x_train = x_train.reshape(-1,6,5,1)
x_test = x_test.reshape(-1,6,5,1)
from tensorflow.keras.layers import Dropout, Flatten, Conv2D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import time

model = Sequential()
model.add(Conv2D(64, (2,2), strides=1, input_shape=(6,5,1), padding='same'))
model.add(Conv2D(64, (2,2), padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2),activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss', mode='min',
    patience=50, restore_best_weights=True, verbose=1
)

date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
path1 = './_save/keras41/06cancer/'
filename = '({epoch:04d}-{val_loss:.4f}).hdf5'
filepath = ''.join([path1, 'k41_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='min',
    save_best_only=True, filepath=filepath,
    verbose=1
)

s_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=10000, batch_size=64,
    verbose=2, validation_split=0.2,
    callbacks=[es, mcp]
)
e_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
# print(y_predict[:10])
y_predict = np.round(y_predict) # pyhon의 반올림

print('#####06#####')
print('[BCE](소숫점 4번째 자리까지 표시) :', round(results[0], 4))      # [BCE](소숫점 4번째 자리까지 표시) : 0.1399
print('[ACC](소숫점 4번째 자리까지 표시) :', round(results[1], 4))      # [ACC](소숫점 4번째 자리까지 표시) : 0.9415   

from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
accuracy_score = np.round(accuracy_score, 4)
print("acc_score :", accuracy_score)
print('time :', np.round(e_time - s_time, 1), 'sec')
# print('걸린 시간 :', round(end_time - start_time, 2), '초')

# [BCE](소숫점 4번째 자리까지 표시) : 0.0978
# [ACC](소숫점 4번째 자리까지 표시) : 0.9766
# acc_score : 0.9766
# 걸린 시간 : 4.66 초


# MinMaxScaler
# [BCE](소숫점 4번째 자리까지 표시) : 0.0705
# [ACC](소숫점 4번째 자리까지 표시) : 0.9766
# acc_score : 0.9766

# MaxAbsScaler
# [BCE](소숫점 4번째 자리까지 표시) : 0.0671
# [ACC](소숫점 4번째 자리까지 표시) : 0.9649
# acc_score : 0.9649

# StandardScaler
# [BCE](소숫점 4번째 자리까지 표시) : 0.1306
# [ACC](소숫점 4번째 자리까지 표시) : 0.9825
# acc_score : 0.9825

# RobustScaler
# [BCE](소숫점 4번째 자리까지 표시) : 0.0761
# [ACC](소숫점 4번째 자리까지 표시) : 0.9766
# acc_score : 0.9766

# CNN
# [BCE](소숫점 4번째 자리까지 표시) : 0.4118
# [ACC](소숫점 4번째 자리까지 표시) : 0.9415
# acc_score : 0.9415
# time : 9.2 sec