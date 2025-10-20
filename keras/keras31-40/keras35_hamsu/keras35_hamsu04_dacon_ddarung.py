from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

path ='./_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv')

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
print(x.shape)  #(1459, 9)
print(y.shape)  #(1459,)

import random
r = 7275 #random.randint(1,10000)     # 7275, 208, 6544, 1850, 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=r
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

from tensorflow.keras.layers import Dropout


# model = Sequential()
# model.add(Dense(100, input_dim=9, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1, activation='linear'))

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
input1= Input(shape=(9,))
dense1= Dense(100, name='d1')(input1)
drop1 = Dropout(0.5)(dense1)
dense2= Dense(100)(drop1)
drop2 = Dropout(0.4)(dense2)
dense3= Dense(100)(drop2)
drop3 = Dropout(0.3)(dense3)
dense4= Dense(100)(drop3)
dense5= Dense(100)(dense4)
output1= Dense(1)(dense5)
model = Model(inputs=input1, outputs =output1)


es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=100,
    restore_best_weights=True
)


model.compile(loss='mse', optimizer='adam')
str_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000000, batch_size=21,
          verbose=2, validation_split=0.2,
          callbacks=[es],
          )
end_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)
# print('Random :', r)
print('RMSE :', rmse)
print('R2 :', r2)
# print('time :', end_time - str_time, '초')

# y_submit = model.predict(test_csv)
# submission_csv['count'] = y_submit
# submission_csv.to_csv(path + 'submission_0526_2.csv', index=False)


# Random : 7275
# RMSE : 42.97903854097076
# R2 : 0.744493249355315
# time : 25.67926526069641 초

# MinMaxScaler
# RMSE : 39.71161761049514
# R2 : 0.7818656088957502

# MaxAbsScaler
# RMSE : 39.81452206158185
# R2 : 0.7807336392832076

# StandardScaler
# RMSE : 43.893867804732984
# R2 : 0.733500303121263

# RobustScaler
# RMSE : 45.13279694802398
# R2 : 0.7182437680788002

# RMSE : 42.19618978790843
# R2 : 0.7537163742095903