# dacon, 데이터 파일 별도
# https://dacon.io/competitions/official/236068/leaderboard

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = './Study25/_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_na_columns] = x[zero_na_columns].replace(0, np.nan)
x = x.fillna(x.mean())
y = train_csv['Outcome']

# r = 55
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.9, shuffle=True, random_state=r
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
# x_train = x_train.reshape(-1,2,2,2)
# x_test = x_test.reshape(-1,2,2,2)
# # exit()
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
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss', mode='min',
#     patience=50, restore_best_weights=True, verbose=1
# )

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# path1 = './_save/keras41/07당뇨/'
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

# results = model.evaluate(x_test, y_test)
# y_predict = model.predict(x_test)
# y_predict = np.round(y_predict)
# accuracy_score = accuracy_score(y_test, y_predict)

# # print(y_predict)
# # print('RanNo. :', r)
# print('CNN 07')
# print('Loss   :', round(results[0], 4))
# print('Acc    :', round(accuracy_score, 4))
# print('time :', np.round(e_time - s_time, 1), 'sec')

# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)
# submission_csv['Outcome'] = y_submit
# submission_csv.to_csv(path + 'submission_1.csv', index=False)


# Loss   : 0.5203
# Acc    : 0.7879

# MinMaxScaler
# Loss   : 0.3736
# Acc    : 0.8333

# MaxAbsScaler
# Loss   : 0.5119
# Acc    : 0.8485

# StandardScaler
# Loss   : 0.4167
# Acc    : 0.8485

# RobustScaler
# Loss   : 0.4021
# Acc    : 0.8182

# CNN 07
# Loss   : 0.4053
# Acc    : 0.8333
# time : 6.0 sec

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
n_split=5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)
model = HistGradientBoostingClassifier()
scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))
# acc : [0.74045802 0.75572519 0.73076923 0.69230769 0.74615385] 
# 평균 acc : 0.7331