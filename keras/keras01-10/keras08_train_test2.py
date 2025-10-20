from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10]).T
y = np.array([1,2,3,4,5,6,7,8,9,10]).T

x_train = x[:7]
# x_train = x[0:7]
y_train = y[:7]
# y_train = y[0:7]
print(x_train)  # [1 2 3 4 5 6 7]
print(y_train)  # [1 2 3 4 5 6 7]
print(x_train.shape, y_train.shape) #(7,) (7,)

x_test = x[7:]
# x_test = x[7:10]
y_test = y[7:]
# y_test = y[7:10]
print(x_test)   # [ 8  9 10]
print(y_test)   # [ 8  9 10]
print(x_test.shape, y_test.shape)   #(3,) (3,)

# x_train = np.array([1,2,3,4,5,6,7])
# y_train = np.array([1,2,3,4,5,6,7])

# x_test  = np.array([8,9,10])
# y_test  = np.array([8,9,10])

# exit()

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([11])

print('#####################')
print('loss :', loss)
print('예측값 :', result)

# #####################
# loss : 1.5162554518610705e-05
# 예측값 : [[10.994236]]