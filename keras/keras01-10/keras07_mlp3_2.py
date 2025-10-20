from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)]) #(3,10)
y = np.array([[ 1, 2, 3, 4, 5, 6, 7, 8, 9,10],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
              [ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]) #(3,10)
x = x.T
y = y.T
print(x.shape)  #(10, 3)
print(y.shape)  #(10, 3)

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

n = [[10,31,211],
     [11,32,212]]
loss = model.evaluate(x, y)
results = model.predict(n)
print('################')
print('loss :',loss)
print(n,'예측값 :',results)

################
# loss : 1.6967149907287649e-09
# [[10, 31, 211], [11, 32, 212]] 예측값 :
# [[ 1.1000053e+01,  3.4673954e-05, -9.9992305e-01],
#  [ 1.2000074e+01, -9.9997807e-01, -1.9999299e+00]]