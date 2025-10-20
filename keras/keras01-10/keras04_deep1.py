from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

#2. 모델구성
model = Sequential()

model.add(Dense(100, input_dim=1))   #파라미터
model.add(Dense(100, input_dim=100)) #파라미터
model.add(Dense(100, input_dim=100)) #파라미터
model.add(Dense(1, input_dim=100))   #파라미터

epochs = 300  #고정
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') #파라미터
model.fit(x, y, epochs=epochs)

print('#############################')
#4. 평가, 예측
loss = model.evaluate(x,y)
print('epochs : ', epochs)
print('loss : ', loss)
result = model.predict(np.array([6]))
print('6의 예측값 : ',result)

















