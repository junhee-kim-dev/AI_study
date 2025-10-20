from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
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


r = 42 #random.randint(1, 10000)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=r,
)

print(x_train.shape, x_test.shape)  # (398, 30) (171, 30)
print(y_train.shape, y_test.shape)  # (398,   ) (171,   )

#2. 모델구성
model = Sequential()
model.add(Dense(128, input_dim=30, activation='relu'))  # activation='relu' output layer만 아니면 사용해라.
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))    

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')               # mse는 거리를 기준으로 하는데 sigmoid에는 의미없는 거리일 뿐임.
model.compile(loss='binary_crossentropy', optimizer='adam', # 이진 분류에는 100% binary_crossentropy를 쓴다.
              metrics=['acc'],
              )

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True
)


import datetime
date = datetime.datetime.now()
print(date)                     
print(type(date))               
date = date.strftime('%m%d_%H%M')
print(date)                    
print(type(date))              

path = './_save/keras28_mcp/06_cancer/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k28_', date, '_', filename])

from tensorflow.keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto',
    save_best_only=True, 
    filepath=filepath
)

start_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=100000, batch_size=32,
    verbose=2, validation_split=0.2,
    callbacks=[es, mcp],
)
end_time = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
# print(y_predict[:10])
y_predict = np.round(y_predict) # pyhon의 반올림

print('[BCE](소숫점 4번째 자리까지 표시) :', round(results[0], 4))      # [BCE](소숫점 4번째 자리까지 표시) : 0.1399
print('[ACC](소숫점 4번째 자리까지 표시) :', round(results[1], 4))      # [ACC](소숫점 4번째 자리까지 표시) : 0.9415   

from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, y_predict)
accuracy_score = np.round(accuracy_score, 4)
print("acc_score :", accuracy_score)
print('걸린 시간 :', round(end_time - start_time, 2), '초')

# [BCE](소숫점 4번째 자리까지 표시) : 0.0978
# [ACC](소숫점 4번째 자리까지 표시) : 0.9766
# acc_score : 0.9766
# 걸린 시간 : 4.66 초