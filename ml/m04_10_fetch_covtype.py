import ssl

# SSL 인증서 문제 해결
ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_covtype
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, Dropout, Input
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import numpy as np
import pandas as pd

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# y = y.reshape(-1, 1)
# ohe = OneHotEncoder(sparse=False)
# y = ohe.fit_transform(y)
# # print(y)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=50, stratify=y
)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

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

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# #2. 모델구성
# model = Sequential()
# model.add(Dense(64, input_dim=54, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(7, activation='softmax'))

# input1= Input(shape=(54,))
# dense1= Dense(64)(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2= Dense(128)(drop1)
# drop2 = Dropout(0.4)(dense2)
# dense3= Dense(128)(drop2)
# drop3 = Dropout(0.3)(dense3)
# dense4= Dense(128)(drop3)
# dense5= Dense(128)(dense4)
# dense6= Dense(64)(dense5)
# dense7= Dense(32)(dense6)
# output= Dense(7)(dense7)
# model = Model(inputs=input1, outputs=output)



# #3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(
#     monitor='val_loss', mode='min', patience=10, restore_best_weights=True
# )

# model.fit(x_train, y_train, epochs=10000, batch_size=256,
#           validation_split=0.2, callbacks=[es], verbose=2)

# #4. 평가, 예측
# results = model.evaluate(x_test, y_test)
# print('loss : ', results[0])
# print('acc : ', results[1])
# y_predict = model.predict(x_test)
# y_round = np.round(y_predict)
# f1 = f1_score(y_test, y_round, average='macro')
# print('f1 : ', f1)


# loss :  0.16358691453933716
# acc :  0.9360687732696533
# f1 :  0.9026860016864566

# MinMaxScaler
# loss :  0.1545630395412445
# acc :  0.9390290975570679
# f1 :  0.8996625504436924

# MaxAbsScaler
# loss :  0.15300233662128448
# acc :  0.9401650428771973
# f1 :  0.9064630078383792

# StandardScaler
# loss :  0.12790009379386902
# acc :  0.9528153538703918
# f1 :  0.9210544034332827

# RobustScaler
# loss :  0.13307805359363556
# acc :  0.952393651008606
# f1 :  0.9195221774313842

# loss :  0.45417284965515137
# acc :  0.8136537075042725
# f1 :  0.5702000523764569

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
models = [LinearSVC(C=0.3), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(f'{model} :', results)
    
# LinearSVC(C=0.3) : 0.7142930905398311
# LogisticRegression() : 0.7259192964037073
# DecisionTreeClassifier() : 0.9381685498653219
# RandomForestClassifier() : 0.9559822035575674