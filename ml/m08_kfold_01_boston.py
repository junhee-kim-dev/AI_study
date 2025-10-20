# 27-3 카피

# import sklearn as sk
# print(sk.__version__)       #0.24.2

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import time

#1. 데이터
from sklearn.datasets import load_boston
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True,
    random_state=333,
)

from sklearn.preprocessing import RobustScaler

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

# print(x_train.shape, x_test.shape)  #(404, 13) (102, 13)
# exit()
# x_train = x_train.reshape(-1, )

# x_train = x_train.reshape(-1,13,1,1)
# x_test = x_test.reshape(-1,13,1,1)

# #2. 모델 구성
# # model = Sequential()
# # model.add(Conv2D(64, (3,1), strides=1, input_shape=(13,1,1)))
# # model.add(Conv2D(64, (3,1)))
# # model.add(Dropout(0.2))
# # model.add(Conv2D(32, (3,1),activation='relu'))
# # model.add(Flatten())
# # model.add(Dense(32, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(16, activation='relu'))
# # model.add(Dense(1, activation='linear'))
# # from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

# #3. 컴파일, 훈련

# model.compile(loss='mse', optimizer='adam', metrics=['acc'])

# from keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(
#     monitor='val_loss', mode='min',
#     patience=50, restore_best_weights=True, verbose=1
# )

# import datetime

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# path1 = './_save/keras41/01boston/'
# filename = '({epoch:04d}-{val_loss:.4f}).hdf5'
# filepath = ''.join([path1, 'k41_', date, '_', filename])

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='min',
#     save_best_only=True, filepath=filepath,
#     verbose=1
# )

# s_time = time.time()
# hist = model.fit(
#     x_train, y_train, epochs=10000, batch_size=64,
#     verbose=2, validation_split=0.2,
#     callbacks=[es, mcp]
# )
# e_time = time.time()

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# results = model.predict(x_test)
# rmse = np.sqrt(loss[0])
# r2 = r2_score(y_test, results)

# print('######boston#######')
# print('CNN')
# print('RMSE :', rmse)   
# print('R2 :', r2)
# print('time :', np.round(e_time - s_time, 1), 'sec')

# ###################
# RMSE : 4.910753260866537
# R2 : 0.7541216411716923

# MinMaxScaler
# RMSE : 4.909639973231036
# R2 : 0.7542331069473764

# MaxAbsScaler
# RMSE : 5.063856531819747
# R2 : 0.7385510723780746

# StandardScaler
# RMSE : 5.049883344454853
# R2 : 0.7399919833098837

# RobustScaler
# RMSE : 4.89211886567054
# R2 : 0.7559840963384721

# CNN
# RMSE : 4.715837603024059
# R2 : 0.7732528857780434
# time : 10.6 sec

from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
n_split=5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
model = MLPClassifier()
scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))