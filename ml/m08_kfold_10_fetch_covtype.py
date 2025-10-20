# import ssl

# # SSL 인증서 문제 해결
# ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import numpy as np
import pandas as pd

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# y = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)
# # print(y)

# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=50, stratify=y
# )

# from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

# # scaler = MinMaxScaler()
# # scaler.fit(x_train)
# # x_train = scaler.transform(x_train)
# # x_test = scaler.transform(x_test)

# # scaler = MaxAbsScaler()
# # scaler.fit(x_train)
# # x_train = scaler.transform(x_train)
# # x_test = scaler.transform(x_test)

# # scaler = StandardScaler()
# # scaler.fit(x_train)
# # x_train = scaler.transform(x_train)
# # x_test = scaler.transform(x_test)

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
# exit()
# x_train = x_train.reshape(-1,2,2,2)
# x_test = x_test.reshape(-1,2,2,2)
# from tensorflow.keras.layers import Dropout, Flatten, Conv2D
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# import time

# model = Sequential()
# model.add(Conv2D(64, (2,2), strides=1, input_shape=(2,2,2), padding='same'))
# model.add(Conv2D(64, (2,2), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2),activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='softmax'))

# #3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss', mode='min',
#     patience=50, restore_best_weights=True, verbose=1
# )

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# path1 = './_save/keras41/10fetch/'
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
# results = model.evaluate(x_test, y_test)
# print('loss : ', results[0])
# print('acc : ', results[1])
# y_predict = model.predict(x_test)
# y_round = np.round(y_predict)
# f1 = f1_score(y_test, y_round, average='macro')

# print('CNN')
# print('f1 : ', f1)
# print('time :', np.round(e_time - s_time, 1), 'sec')


# loss :  0.16358691453933716
# acc :  0.9360687732696533
# f1 :  0.9026860016864566

# MinMaxScaler
# loss :  0.1545630395412445
# acc :  0.9390290975570679
# f1 :  0.8996625504436924

# MaxAbsScaler
# loss :  0.15300233662128448
# acc :  0.9401650428771973
# f1 :  0.9064630078383792

# StandardScaler
# loss :  0.12790009379386902
# acc :  0.9528153538703918
# f1 :  0.9210544034332827

# RobustScaler
# loss :  0.13307805359363556
# acc :  0.952393651008606
# f1 :  0.9195221774313842

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
n_split=5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)
model = HistGradientBoostingClassifier()
scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))

# acc : [0.80596026 0.780195   0.84246399 0.77857524 0.77693155] 
# 평균 acc : 0.7968