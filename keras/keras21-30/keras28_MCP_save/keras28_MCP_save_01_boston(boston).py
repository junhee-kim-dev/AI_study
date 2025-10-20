# 27-3 카피

import sklearn as sk
print(sk.__version__)       #0.24.2

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time

#1. 데이터
from sklearn.datasets import load_boston
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True,
    random_state=333,
)

mms = MinMaxScaler()
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=13))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(13))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(
    monitor='val_loss', mode='min', patience=20,
    restore_best_weights=True,
)

################# mcp 세이브 파일명 만들기 ##################
import datetime
date = datetime.datetime.now()
print(date)                     
print(type(date))               
date = date.strftime('%m%d_%H%M')
print(date)                    
print(type(date))              

path = './_save/keras28_mcp/01_boston/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k28_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto',
    save_best_only=True, 
    filepath=filepath
)

hist = model.fit(
    x_train, y_train, 
    epochs=1000000, batch_size=1, 
    verbose=2, validation_split=0.2,
    callbacks=[es, mcp],
)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('###################')
print('RMSE :', rmse)   
print('R2 :', r2)

# ###################
# RMSE : 4.910753260866537
# R2 : 0.7541216411716923