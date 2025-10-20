import ssl

# # SSL 인증서 문제 해결
ssl._create_default_https_context = ssl._create_unverified_context


import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler

#1. 데이터
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, shuffle=True, random_state=304)


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


#2. 모델 구성
model = Sequential()
model.add(Dense(40, input_dim=8))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

import time
s_time = time.time()

hist = model.fit(
        x_train, y_train, epochs=100, 
        batch_size=32, verbose=2, validation_split=0.2,
        callbacks=[],
        )
e_time = time.time()


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)

print('############')
print('RMSE :', rmse)
print('R2 :', r2)


# ############
# RMSE : 0.8331345194964307
# R2 : 0.4864231795103535

# MinMaxScaler
# RMSE : 0.7526945625543631
# R2 : 0.5808082628321776

# MaxAbsScaler
# RMSE : 0.7775688249564648
# R2 : 0.5526444431455901

# StandardScaler
# RMSE : 0.7471945903085894
# R2 : 0.5869119858631187

# RobustScaler
# RMSE : 0.7768150762497256
# R2 : 0.553511325226224

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print(gpus) # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')
print('time:', np.round(e_time - s_time, 1), 'sec')

# GPU 있다~
# time: 105.6 sec

# GPU 없다~
# time: 32.7 sec