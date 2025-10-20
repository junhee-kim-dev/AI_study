import numpy as np
import pandas as pd
import sklearn as sk

from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(datasets.feature_names)

print(x.shape)  # (178, 13)
print(y.shape)  # (178,)
print(np.unique(y, return_counts=True))

ohe = OneHotEncoder(sparse=False)
y = y.reshape(-1, 1)
y = ohe.fit_transform(y)
print(type(x))  # <class 'numpy.ndarray'>
print(type(y))  # <class 'numpy.ndarray'>


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=337, stratify=y # stratify는 x와 y를 균등한 비율로 분배
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

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

input1= Input(shape=(13,))
dense1= Dense(64)(input1)
drop1 = Dropout(0.5)(dense1)
dense2= Dense(128)(drop1)
drop2 = Dropout(0.4)(dense2)
dense3= Dense(128)(drop2)
drop3 = Dropout(0.3)(dense3)
dense4= Dense(128)(drop3)
dense5= Dense(128)(dense4)
dense6= Dense(128)(dense5)
dense7= Dense(64)(dense6)
output1= Dense(3)(dense7)
model = Model(inputs=input1, outputs=output1)



#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)


model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2,
          callbacks=[es], verbose=2)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])
y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
f1 = f1_score(y_test, y_predict, average='macro')
print('f1_score : ', f1)

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

# loss :  0.060566410422325134
# acc :  0.9444444179534912
# f1_score :  0.942857142857143