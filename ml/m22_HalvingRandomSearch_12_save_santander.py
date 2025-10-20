# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import datetime
import time

path = './Study25/_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# print(x.shape)  #(200000, 200)
# print(y.shape)  #(200000,)

# # y = pd.get_dummies(y)
# # ohe = OneHotEncoder(sparse=True)
# # y = ohe.fit_transform(y)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, shuffle=True, random_state=123, stratify=y
# )

# ms = MaxAbsScaler()
# ms.fit(x_train)
# x_train = ms.transform(x_train)
# x_test = ms.transform(x_test)
# test_csv = ms.transform(test_csv)

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
# # exit()
# x_train = x_train.reshape(-1,10,10,2)
# x_test = x_test.reshape(-1,10,10,2)
# from tensorflow.keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# import time

# model = Sequential()
# model.add(Conv2D(64, (2,2), strides=1, input_shape=(10,10,2), padding='same'))
# model.add(MaxPooling2D())
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
# path1 = './_save/keras41/12santander/'
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
# results = np.round(results)
# f1 = f1_score(y_test, results)

# # y_submit = model.predict(test_csv)
# # sub_csv['target'] = y_submit
# # filename1 = ''.join(['submission_', date,'.csv'])
# # sub_csv.to_csv(path+ filename1)
# # print('File :', filename1)
# # import tensorflow as tf

# # gpus = tf.config.list_physical_devices('GPU')
# # # print(gpus) # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# # if gpus:
# #     print('GPU 있다~')
# # else:
# #     print('GPU 없다~')
# print('CNN 12')
# print('F1  :', f1)
# print('time:', np.round(e_time - s_time, 2), 'sec')


# File : submission_0608_2242.csv
# GPU 없다~
# F1  : 0.362078599366735
# time: 259.94 sec


# File : submission_0608_2248.csv
# GPU 있다~
# F1  : 0.4329590488771466
# time: 220.57 sec

# CNN 12
# F1  : 0.35077739855896856
# time: 498.55 sec
# s_time = time.time()
# from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# n_split=5
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)
# model = HistGradientBoostingClassifier()
# scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
# e_time = time.time()
# print(e_time - s_time)
# print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))

# acc : [0.907425 0.90765  0.908375 0.907375 0.907475] 
# 평균 acc : 0.9077

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
# print('   test acc :', acc)

#         acc : [0.86128923 0.86168039 0.85639285]
# average acc : 0.85979
#    test acc : 0.885725


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


# import math
# print(x_train.shape)
# factor = 3
# n_iterations = 4
# min_resources = max(1, math.floor(x_train.shape[0] // (factor ** (n_iterations - 1))))

# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import HalvingGridSearchCV

# model = HalvingGridSearchCV(xgb, parameters, cv=kfold,
#                         verbose=1,
#                         refit=True,
#                         n_jobs=18,
#                         factor=factor,               # 배율 (min_resources * 3) *3 ...
#                         min_resources=min_resources,       # 최소 훈련량
#                         random_state=333
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
acc = accuracy_score(y_test, y_pred)
print('   accuracy_score :', acc)
print('     running_time :', np.round(e_time - s_time, 3), 'sec')

#       best_params : {'learning_rate': 0.1, 'min_child_weight': 10}
#        best_score : 0.907585560243788
#  model_best_score : 0.906975
#    accuracy_score : 0.906975
#      running_time : 175.246 sec

# import pandas as pd

# best_csv = pd.DataFrame(model.cv_results_).sort_values(['rank_test_score'], ascending=True)

# path = './Study25/_save/ml/'
# best_csv.to_csv(path + 'm15/cv_results_12.csv', index=False)

import joblib
path = './Study25/_save/ml/m22/'
joblib.dump(model.best_estimator_, path + 'm22_best_model_12.joblib')