from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint

path ='./Study25/_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv')

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
print(x.shape)  #(1459, 9)
print(y.shape)  #(1459,)

# import random
# r = 7275 #random.randint(1,10000)     #7275, 208, 6544, 1850, 

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, shuffle=True, random_state=r
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
# # exit()
# x_train = x_train.reshape(-1,3,3,1)
# x_test = x_test.reshape(-1,3,3,1)
# from keras.layers import Dropout, Flatten, Conv2D
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# import time

# model = Sequential()
# model.add(Conv2D(64, (2,2), strides=1, input_shape=(3,3,1), padding='same'))
# model.add(Conv2D(64, (2,2), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2),activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='linear'))

# #3. 컴파일, 훈련

# model.compile(loss='mse', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss', mode='min',
#     patience=50, restore_best_weights=True, verbose=1
# )

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# path1 = './_save/keras41/04ddarung/'
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

# loss = model.evaluate(x_test, y_test)
# results = model.predict(x_test)
# rmse = np.sqrt(loss[0])
# r2 = r2_score(y_test, results)
# # print('Random :', r)
# print('#####04####')
# print('CNN')
# print('RMSE :', rmse)
# print('R2 :', r2)
# print('time :', np.round(e_time - s_time, 1), 'sec')
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

# CNN
# RMSE : 46.33145013362279
# R2 : 0.7030791057465402
# time : 18.6 sec

# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# n_split=5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
# model = HistGradientBoostingRegressor()
# scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
# print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))

# # acc : [0.79152438 0.75248885 0.76504642 0.7960695  0.83243112] 
# # 평균 acc : 0.7875

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=42, train_size=0.8)

ss = StandardScaler()
ss.fit(x_train)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

n_split = 3
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

#2. 모델구성
model = HistGradientBoostingRegressor()

#3. 훈련
score = cross_val_score(model, x_train, y_train, cv=kfold)
print('        r2 :', score)
print('average r2 :', round(np.mean(score), 5))
results = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test, results)
print('   test r2 :', r2)

#         r2 : [0.77103941 0.74480872 0.78456196]
# average r2 : 0.7668
#    test r2 : 0.6817360425056822
