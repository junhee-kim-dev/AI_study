# https://www.kaggle.com/competitions/playground-series-s4e1/submissions

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import f1_score
from keras.layers import BatchNormalization
from keras.layers import Dropout
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = './Study25/_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

train_csv[['Tenure', 'Balance']] = train_csv[['Tenure', 'Balance']].replace(0, np.nan)
train_csv = train_csv.fillna(train_csv.mean())

test_csv[['Tenure', 'Balance']] = test_csv[['Tenure', 'Balance']].replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

oe = OrdinalEncoder()       # 이렇게 정의 하는 것을 인스턴스화 한다고 함
oe.fit(train_csv[['Geography', 'Gender']])
train_csv[['Geography', 'Gender']] = oe.transform(train_csv[['Geography', 'Gender']])
test_csv[['Geography', 'Gender']] = oe.transform(test_csv[['Geography', 'Gender']])

train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, shuffle=True, random_state=123
# )


# from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# # scaler = MaxAbsScaler()
# # scaler.fit(x_train)
# # x_train = scaler.transform(x_train)
# # x_test = scaler.transform(x_test)

# # scaler = StandardScaler()
# # scaler.fit(x_train)
# # x_train = scaler.transform(x_train)
# # x_test = scaler.transform(x_test)

# # scaler = RobustScaler()
# # scaler.fit(x_train)
# # x_train = scaler.transform(x_train)
# # x_test = scaler.transform(x_test)


# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
# x_train = x_train.reshape(-1,5,2,1)
# x_test = x_test.reshape(-1,5,2,1)
# # exit()
# from tensorflow.keras.layers import Dropout, Flatten, Conv2D
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
# model.add(Dense(1, activation='sigmoid'))

# #3. 컴파일, 훈련

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss', mode='min',
#     patience=50, restore_best_weights=True, verbose=1
# )

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# path1 = './_save/keras41/08bank/'
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
# acc = accuracy_score(y_test, results)
# f1 = f1_score(y_test, results)

# print('CNN 08')
# print('Loss :', loss[0])
# print('Acc  :', acc)
# print('F1   :', f1)
# print('time :', np.round(e_time - s_time, 1), 'sec')

# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)
# submission_csv['Exited'] = y_submit
# submission_csv.to_csv(path + 'submission_0527_1.csv', index=False)


# Loss : 0.3281714916229248
# Acc  : 0.862483715575484
# F1   : 0.6295601077287195


# CNN 08
# Loss : 0.33338749408721924
# Acc  : 0.86021147029418
# F1   : 0.6185515873015873
# time : 759.9 sec

# from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# n_split=5
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)
# model = HistGradientBoostingClassifier()
# scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
# print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))

# acc : [0.86339261 0.86872482 0.86405914 0.86557397 0.86363085] 
# 평균 acc : 0.8651


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

#         acc : [0.85850621 0.86289168 0.86405054]
# average acc : 0.86182
#     test acc: 0.857302996334111



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

import joblib
path = './Study25/_save/ml/m20/'
model = joblib.load(path + 'm20_best_model_08.joblib')
print(model)
print(type(model))
'''
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              feature_weights=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=None,
              max_leaves=None, min_child_weight=3, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=None,
              n_jobs=None, num_parallel_tree=None, ...)
<class 'xgboost.sklearn.XGBClassifier'>
'''
'''
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
'''
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('   accuracy_score :', acc)
# print('     running_time :', np.round(e_time - s_time, 3), 'sec')

#       best_params : {'learning_rate': 0.1, 'min_child_weight': 3}
#        best_score : 0.8659440729208299
#  model_best_score : 0.8651195201017966
#    accuracy_score : 0.8651195201017966
#      running_time : 48.375 sec

# import pandas as pd

# best_csv = pd.DataFrame(model.cv_results_).sort_values(['rank_test_score'], ascending=True)

# path = './Study25/_save/ml/'
# best_csv.to_csv(path + 'm15/cv_results_08.csv', index=False)