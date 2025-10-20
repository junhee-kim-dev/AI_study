import tensorflow as tf
# print(tf.__version__)
import numpy as np
# print(np.__version__)

from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target
print(x.shape)  #(20640, 8)
print(y.shape)  #(20640, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, shuffle=True, random_state=304)

#2. 모델 구성
model = Sequential()
model.add(Dense(40, input_dim=8))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
b = 50
e = 200
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=e, batch_size=b, verbose=1, validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)

print('############')
print('batch_size :', b)
print('epochs :', e)
print('loss :', loss)
# print('예측값 :', results)
print('RMSE :', rmse)
print('R2 :', r2)

#전
# ############
# train_size=0.9, test_size=0.1
# random_state=304
# batch_size : 50
# epochs : 200
# loss : 0.6470549702644348
# RMSE : 0.8043972508998458
# R2 : 0.521241705198811

#후
# ############
# batch_size : 50
# epochs : 200
# loss : 0.6729736924171448
# RMSE : 0.820349715352489
# R2 : 0.5020643515836155