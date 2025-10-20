import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13, 8,14,15, 9, 6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
reults = model.predict(x_test)

print('###########')
print('loss :', loss)
print('[x]의 예측값 :', reults)

from sklearn.metrics import r2_score, mean_squared_error

def RMSE(y_test, y_predict):  # def(define) : 함수
  return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, reults)
print('RMSE :', rmse)