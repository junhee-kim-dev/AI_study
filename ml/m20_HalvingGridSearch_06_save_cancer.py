from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
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


# r = 42 #random.randint(1, 10000)
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.7, shuffle=True, random_state=r,
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
# x_train = x_train.reshape(-1,6,5,1)
# x_test = x_test.reshape(-1,6,5,1)
# from tensorflow.keras.layers import Dropout, Flatten, Conv2D
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# import time

# model = Sequential()
# model.add(Conv2D(64, (2,2), strides=1, input_shape=(6,5,1), padding='same'))
# model.add(Conv2D(64, (2,2), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2),activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss', mode='min',
#     patience=50, restore_best_weights=True, verbose=1
# )

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# path1 = './_save/keras41/06cancer/'
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
# y_predict = model.predict(x_test)
# # print(y_predict[:10])
# y_predict = np.round(y_predict) # pyhon의 반올림

# print('#####06#####')
# print('[BCE](소숫점 4번째 자리까지 표시) :', round(results[0], 4))      # [BCE](소숫점 4번째 자리까지 표시) : 0.1399
# print('[ACC](소숫점 4번째 자리까지 표시) :', round(results[1], 4))      # [ACC](소숫점 4번째 자리까지 표시) : 0.9415   

# from sklearn.metrics import accuracy_score
# accuracy_score = accuracy_score(y_test, y_predict)
# accuracy_score = np.round(accuracy_score, 4)
# print("acc_score :", accuracy_score)
# print('time :', np.round(e_time - s_time, 1), 'sec')
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

# from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# n_split=5
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)
# model = HistGradientBoostingClassifier()
# scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
# print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))

# acc : [0.95614035 0.97368421 0.95614035 0.99122807 0.92920354] 
# 평균 acc : 0.9613


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
#         acc : [0.98684211 0.96710526 0.98675497]
# average acc : 0.98023
#     test acc: 0.9385964912280702



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
# model = GridSearchCV(xgb, parameters, cv=kfold,
#                      verbose=1,
#                      refit=True,
#                      n_jobs=18,
#                      )

# from sklearn.model_selection import RandomizedSearchCV
# model = RandomizedSearchCV(xgb, parameters, cv=kfold,
#                      verbose=1,
#                      refit=True,
#                      n_jobs=18,
#                      random_state=333,
#                      )


import math
print(x_train.shape)
factor = 3
n_iterations = 3
min_resources = max(1, math.floor(x_train.shape[0] // (factor ** (n_iterations - 1))))

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

model = HalvingGridSearchCV(xgb, parameters, cv=kfold,
                        verbose=1,
                        refit=True,
                        n_jobs=18,
                        factor=factor,               # 배율 (min_resources * 3) *3 ...
                        min_resources=min_resources,       # 최소 훈련량
                        random_state=333
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
#        best_score : 0.9736263736263735
#  model_best_score : 0.9473684210526315
#    accuracy_score : 0.9473684210526315
#      running_time : 4.217 sec

# import pandas as pd

# best_csv = pd.DataFrame(model.cv_results_).sort_values(['rank_test_score'], ascending=True)

# path = './Study25/_save/ml/'
# best_csv.to_csv(path + 'm15/cv_results_06.csv', index=False)

import joblib
path = './Study25/_save/ml/m20/'
joblib.dump(model.best_estimator_, path + 'm20_best_model_06.joblib')

#       best_params : {'min_child_weight': 2, 'learning_rate': 0.1}
#        best_score : 0.9736263736263735
#  model_best_score : 0.9649122807017544
#    accuracy_score : 0.9649122807017544
#      running_time : 3.582 sec

#       best_params : {'learning_rate': 0.1, 'max_depth': 12}
#        best_score : 0.971111111111111
#  model_best_score : 0.9473684210526315
#    accuracy_score : 0.9473684210526315
#      running_time : 3.227 sec


