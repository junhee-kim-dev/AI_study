from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([range(10), range(21,31), range(201,211)])
y = np.array([[ 1, 2, 3, 4, 5, 6, 7, 8, 9,10],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
x = x.T
y = y.T
print(x.shape)

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

n = [[10,31,211],
     [11,32,212]]
loss = model.evaluate(x, y)
results = model.predict(n)
print('################')
print('loss :',loss)
print(n,'예측값 :',results)

# ################
# loss : 0.000000011345157702180586
# [[10, 31, 211], 
#  [11, 32, 212]]의 예측값 :
# [[ 10.999782, -0.000215],
#  [ 11.999760, -1.000241]]