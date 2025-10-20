from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
import numpy as np
import pandas as pd
import datetime
import time

path = './Study25/_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')

# print(train_csv.shape) #(61878, 94)
# print(test_csv.shape)  #(144368, 93)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# y = pd.get_dummies(y)
# # print(y.shape)  #(61878, 9)
# # exit()

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, shuffle=True, random_state=123
# )

# mas = MaxAbsScaler()
# mas.fit(x_train)
# x_train = mas.transform(x_train)
# x_test = mas.transform(x_test)
# test_csv = mas.transform(test_csv)

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
# # exit()
# x_train = x_train.reshape(-1,31,3,1)
# x_test = x_test.reshape(-1,31,3,1)
# from tensorflow.keras.layers import Dropout, Flatten, Conv2D
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# import time

# model = Sequential()
# model.add(Conv2D(64, (2,2), strides=1, input_shape=(31,3,1)))
# model.add(Conv2D(64, (2,2), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2),activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(9, activation='softmax'))

# #3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss', mode='min',
#     patience=50, restore_best_weights=True, verbose=1
# )

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# path1 = './_save/keras41/13otto/'
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

# loss= model.evaluate(x_test, y_test)
# results = model.predict(x_test)
# results = np.round(results)
# f1 = f1_score(y_test, results, average='macro')

# # y_submit = model.predict(test_csv)
# # y_submit = np.argmax(y_submit, axis=1)
# # sub_csv['target'] = y_submit
# # filename1 = ''.join(['submission_', date, '.csv'])
# # sub_csv.to_csv(path + filename1)
# # print('File: ',filename1)

# # import tensorflow as tf

# # gpus = tf.config.list_physical_devices('GPU')
# # # print(gpus) # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# # if gpus:
# #     print('GPU 있다~')
# # else:
# #     print('GPU 없다~')

# print('F1  :', f1)
# print('time:', np.round(e_time - s_time, 1), 'sec')


# File:  submission_0608_2206.csv
# GPU 없다~
# F1  : 0.769061762965092
# time: 130.2 sec

# File:  submission_0608_2206.csv
# GPU 있다~
# F1  : 0.7721535100421776
# time: 208.4 sec

# CNN
# F1  : 0.7163076220344418
# time: 188.4 sec

# from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# n_split=5
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)
# model = HistGradientBoostingClassifier()
# scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
# print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))

# acc : [0.81633807 0.81577246 0.80825792 0.80856566 0.81115152] 
# 평균 acc : 0.812

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

n_split = 3
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)

#2. 모델구성
model = MLPClassifier()

#3. 훈련
score = cross_val_score(model, x_train, y_train, cv=kfold)
print('        acc :', score)
print('average acc :', round(np.mean(score), 5))
results = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, results)
print('   test acc :', acc)

#         acc : [0.7766196  0.7791043  0.77563636]
# average acc : 0.77712
#    test acc : 0.7553329023917259

