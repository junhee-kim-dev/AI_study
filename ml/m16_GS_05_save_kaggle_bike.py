from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
from keras.callbacks import EarlyStopping

path = './Study25/_data/kaggle/bike-sharing-demand/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['count'], axis=1)
y = train_csv[['count']]
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, shuffle=True, random_state=123
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
# x_train = x_train.reshape(-1,4,2,1)
# x_test = x_test.reshape(-1,4,2,1)
# from keras.layers import Dropout, Flatten, Conv2D
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# import time

# model = Sequential()
# model.add(Conv2D(64, (2,2), strides=1, input_shape=(4,2,1), padding='same'))
# model.add(Conv2D(64, (2,2), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2),activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(2, activation='linear'))

# #3. 컴파일, 훈련

# model.compile(loss='mse', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss', mode='min',
#     patience=50, restore_best_weights=True, verbose=1
# )

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# path1 = './_save/keras41/05bike/'
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

# print('#####05###')
# print('CNN')
# print('RMSE :', rmse)
# print('R2 :', r2)
# print('time :', np.round(e_time - s_time, 1), 'sec')
# print('걸린 시간 :', end_time - srt_time, '초')

# y_submit = model.predict(test_csv)
# test_csv_copy = test_csv.copy()
# test_csv_copy[['casual', 'registered']] = y_submit
# test_csv_copy.to_csv(path + 'new_test_1.csv', index=False)


# RMSE : 95.68644369926442
# R2 : 0.3849638144401447


# MinMaxScaler
# RMSE : 94.75008245378999
# R2 : 0.4237545378319282

# MaxAbsScaler
# RMSE : 95.84237446029549
# R2 : 0.4068468826743233

# StandardScaler
# RMSE : 95.36379767107641
# R2 : 0.41974378420078196

# RobustScaler
# RMSE : 95.36379767107641
# R2 : 0.41974378420078196

# CNN
# RMSE : 96.20101229002219
# R2 : 0.3903037267583487
# time : 50.4 sec

# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# n_split=5
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
# model = HistGradientBoostingRegressor()
# scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
# print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))

# acc : [0.9992764  0.99941511 0.99956179 0.99935931 0.99936808] 
# 평균 acc : 0.9994


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

#         r2 : [0.99938526 0.99936264 0.99935943]
# average r2 : 0.99937
#    test r2 : 0.995921483588714



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

#       best_params : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}
#        best_score : 0.7570526003953303
#  model_best_score : 0.8071493375906007
#    accuracy_score : 0.8071493375906007
#      running_time : 8.407 sec

# import pandas as pd

# best_csv = pd.DataFrame(model.cv_results_).sort_values(['rank_test_score'], ascending=True)

# path = './Study25/_save/ml/'
# best_csv.to_csv(path + 'm15/cv_results_05.csv', index=False)

import joblib
path = './Study25/_save/ml/m16/'
joblib.dump(model.best_estimator_, path + 'm16_best_model_05.joblib')