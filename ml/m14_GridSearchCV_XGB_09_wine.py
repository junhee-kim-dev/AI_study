import numpy as np
import pandas as pd
import sklearn as sk

from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(datasets.feature_names)

# print(x.shape)  # (178, 13)
# print(y.shape)  # (178,)
# print(np.unique(y, return_counts=True))

# ohe = OneHotEncoder(sparse=False)
# y = y.reshape(-1, 1)
# y = ohe.fit_transform(y)
# print(type(x))  # <class 'numpy.ndarray'>
# print(type(y))  # <class 'numpy.ndarray'>


# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=337, stratify=y # stratify는 x와 y를 균등한 비율로 분배
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


# print(x_train.shape, x_test.shape)  #(142, 13) (36, 13)
# print(y_train.shape, y_test.shape)  #(142, 3) (36, 3)
# # exit()
# x_train = x_train.reshape(-1,13,1,1)
# x_test = x_test.reshape(-1,13,1,1)
# from tensorflow.keras.layers import Dropout, Flatten, Conv2D
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# import time

# model = Sequential()
# model.add(Conv2D(64, (2,1), strides=1, input_shape=(13,1,1), padding='same'))
# model.add(Conv2D(64, (2,1), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,1),activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(3, activation='softmax'))

# #3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss', mode='min',
#     patience=50, restore_best_weights=True, verbose=1
# )

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# path1 = './_save/keras41/09wine/'
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
# print('acc : ', results[1])
# y_predict = model.predict(x_test)
# y_predict = np.round(y_predict)
# f1 = f1_score(y_test, y_predict, average='macro')
# print('f1_score : ', f1)
# print('loss : ', results[0])
# print('time :', np.round(e_time - s_time, 1), 'sec')

# loss :  0.10895264893770218
# acc :  0.9444444179534912
# f1_score :  0.945824706694272

# MinMaxScaler
# loss :  0.026520512998104095
# acc :  1.0
# f1_score :  1.0

# MaxAbsScaler
# loss :  0.16276314854621887
# acc :  0.9166666865348816
# f1_score :  0.9181671790367444

# StandardScaler
# loss :  0.23106199502944946
# acc :  0.9444444179534912
# f1_score :  0.942857142857143

# RobustScaler
# loss :  0.13048940896987915
# acc :  0.9722222089767456
# f1_score :  0.9709618874773142

# from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# n_split=5
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)
# model = HistGradientBoostingClassifier()
# scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
# print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))

# acc : [0.97222222 0.88888889 0.97222222 1.         0.97142857] 
# 평균 acc : 0.961

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=42, train_size=0.8, stratify=y)

ss = StandardScaler()
ss.fit(x_train)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

# n_split = 3
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)

# #2. 모델구성
# model = MLPClassifier()

# #3. 훈련
# score = cross_val_score(model, x_train, y_train, cv=kfold)
# print('        acc :', score)
# print('average acc :', round(np.mean(score), 5))
# results = cross_val_predict(model, x_test, y_test, cv=kfold)
# acc = accuracy_score(y_test, results)
# print('    test acc:', acc)

#         acc : [0.9375    1.        0.9787234]
# average acc : 0.97207
#     test acc: 0.9166666666666666


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
import time

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

parameters = [
    {'n_estimators' : [100, 500], 'max_depth' : [6,10,12], 'learning_rate' : [0.1, 0.01, 0.001]},
    {'max_depth' : [6,8,10,12], 'learning_rate' : [0.1, 0.01, 0.001]},
    {'min_child_weight' : [2,3,5,10], 'learning_rate' : [0.1, 0.01, 0.001]}
]  

#2. 모델
xgb = XGBClassifier()
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
acc = accuracy_score(y_test, y_pred)
print('   accuracy_score :', acc)
print('     running_time :', np.round(e_time - s_time, 3), 'sec')

#       best_params : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}
#        best_score : 0.9642857142857144
#  model_best_score : 0.9722222222222222
#    accuracy_score : 0.9722222222222222
#      running_time : 3.217 sec
