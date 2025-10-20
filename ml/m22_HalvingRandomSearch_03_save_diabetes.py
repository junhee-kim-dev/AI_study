from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from keras.callbacks import EarlyStopping

dataset = load_diabetes()
x = dataset.data
y = dataset.target
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, shuffle=True, random_state=50
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
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# x = scaler.fit_transform(x)

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
# # exit()
# x_train = x_train.reshape(-1,5,2,1)
# x_test = x_test.reshape(-1,5,2,1)
# from keras.layers import Dropout, Flatten, Conv2D
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# import time

# model = Sequential()
# model.add(Conv2D(64, (2,2), strides=1, input_shape=(5,2,1), padding='same'))
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
# path1 = './_save/keras41/03diabetes/'
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

# print('######03######')
# print('CNN')
# print('RMSE :', rmse)
# print('R2 :', r2)
# print('time :', np.round(e_time - s_time, 1), 'sec')


# RMSE : 50.469586873866426
# R2 : 0.542062781751149

# MinMaxScaler
# RMSE : 50.73969453556555
# R2 : 0.5371480334293155

# MaxAbsScaler
# RMSE : 49.63791010045271
# R2 : 0.5570309888342313

# StandardScaler
# RMSE : 53.714245920833235
# R2 : 0.4812891184686696

# RobustScaler
# RMSE : 52.68465016758913
# R2 : 0.500983793006675

# CNN
# RMSE : 53.5036460133209
# R2 : 0.48534858314642704
# time : 11.9 sec

# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# n_split=5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
# model = HistGradientBoostingRegressor()
# scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
# print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))

# acc : [0.32241346 0.36710717 0.47591973 0.25545567 0.36580256] 
# 평균 acc : 0.3573

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

#         r2 : [0.44108511 0.45387364 0.39579674]
# average r2 : 0.43025
#    test r2 : 0.2754541010120939


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

# import math
# print(x_train.shape)
# factor = 3
# n_iterations = 3
# min_resources = max(1, math.floor(x_train.shape[0] // (factor ** (n_iterations - 1))))

# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import HalvingGridSearchCV

# model = HalvingGridSearchCV(xgb, parameters, cv=kfold,
#                         verbose=1,
#                         refit=True,
#                         n_jobs=18,
#                         factor=factor,               # 배율 (min_resources * 3) *3 ...
#                         min_resources=min_resources,       # 최소 훈련량
#                         )

import math
print(x_train.shape)
factor = 3
n_iterations = 3
min_resources = max(1, math.floor(x_train.shape[0] // (factor ** (n_iterations - 1))))

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

model = HalvingRandomSearchCV(xgb, parameters, cv=kfold,
                          verbose=1,
                          refit=True,
                          n_jobs=18,
                          factor=factor, 
                          min_resources=min_resources,
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

#       best_params : {'learning_rate': 0.1, 'min_child_weight': 10}
#        best_score : 0.3767878412254125
#  model_best_score : 0.40681174031628675
#    accuracy_score : 0.40681174031628675
#      running_time : 4.501 sec

# import pandas as pd

# best_csv = pd.DataFrame(model.cv_results_).sort_values(['rank_test_score'], ascending=True)

# path = './Study25/_save/ml/'
# best_csv.to_csv(path + 'm15/cv_results_03.csv', index=False)

import joblib
path = './Study25/_save/ml/m22/'
joblib.dump(model.best_estimator_, path + 'm22_best_model_03.joblib')
#       best_params : {'min_child_weight': 10, 'learning_rate': 0.1}
#        best_score : 0.3767878412254125
#  model_best_score : 0.40681174031628675
#    accuracy_score : 0.40681174031628675
#      running_time : 3.932 sec

#       best_params : {'learning_rate': 0.1, 'min_child_weight': 10}
#        best_score : 0.38311797608602555
#  model_best_score : 0.40681174031628675
#    accuracy_score : 0.40681174031628675
#      running_time : 3.145 sec