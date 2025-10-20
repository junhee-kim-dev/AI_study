import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping
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

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# (18576, 8) (2064, 8)
# (18576,) (2064,)
# #2. 모델 구성
# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
# x_train = x_train.reshape(-1,2,2,2)
# x_test = x_test.reshape(-1,2,2,2)
# # exit()
# from keras.layers import Dropout, Flatten, Conv2D
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
# model.add(Dense(1, activation='linear'))

# #3. 컴파일, 훈련

# model.compile(loss='mse', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss', mode='min',
#     patience=50, restore_best_weights=True, verbose=1
# )

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# path1 = './_save/keras41/02california/'
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
# print('time :', np.round(e_time - s_time, 1), 'sec')

# #4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# results = model.predict(x_test)

# def RMSE(a, b) :
#     return np.sqrt(mean_squared_error(a, b))

# rmse = RMSE(y_test, results)
# r2 = r2_score(y_test, results)

# print('####cali#####')
# print('RMSE :', rmse)
# print('R2 :', r2)

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

# CNN
# RMSE : 0.5242390898156843
# R2 : 0.7966547846913941
# time : 158.0 sec

# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# n_split=5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
# model = HistGradientBoostingRegressor()
# scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
# print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))

# acc : [0.82613413 0.8359695  0.82639461 0.84451826 0.84434393] 
# 평균 acc : 0.8355

# from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.metrics import r2_score
# import warnings
# warnings.filterwarnings("ignore")
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, random_state=42, train_size=0.8)

# ss = StandardScaler()
# ss.fit(x_train)
# x_train = ss.transform(x_train)
# x_test = ss.transform(x_test)

# n_split = 3
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

# #2. 모델구성
# model = HistGradientBoostingRegressor()

# #3. 훈련
# score = cross_val_score(model, x_train, y_train, cv=kfold)
# print('        r2 :', score)
# print('average r2 :', round(np.mean(score), 5))
# results = cross_val_predict(model, x_test, y_test, cv=kfold)
# r2 = r2_score(y_test, results)
# print('   test r2 :', r2)

#         r2 : [0.83119187 0.83290788 0.82511233]
# average r2 : 0.82974
#    test r2 : 0.7907807792342676



from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
import time

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

parameters = [
    {'n_estimators' : [100, 500], 'max_depth' : [6,10,12], 'learning_rate' : [0.1, 0.01, 0.001]},
    {'max_depth' : [6,8,10,12], 'learning_rate' : [0.1, 0.01, 0.001]},
    {'min_child_weight' : [2,3,5,10], 'learning_rate' : [0.1, 0.01, 0.001]}
]  

#2. 모델
xgb = XGBRegressor()
model = GridSearchCV(xgb, parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=18,
                     )

#3. 훈련
s_time = time.time()
model.fit(x_train, y_train)
e_time = time.time()

# print('    best_variable :', model.best_estimator_)
print('      best_params :', model.best_params_)

#4. 평가
print('       best_score :', model.best_score_)
print(' model_best_score :', model.score(x_test, y_test))

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('   accuracy_score :', r2)
print('     running_time :', np.round(e_time - s_time, 3), 'sec')

#       best_params : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}
#        best_score : 0.845025942409108
#  model_best_score : 0.8440473231598093
#    accuracy_score : 0.8440473231598093
#      running_time : 58.837 sec

# print(best_csv.columns)

import pandas as pd

best_csv = pd.DataFrame(model.cv_results_).sort_values(['rank_test_score'], ascending=True)

path = './Study25/_save/ml/'
best_csv.to_csv(path + 'm15/cv_results_02.csv', index=False)