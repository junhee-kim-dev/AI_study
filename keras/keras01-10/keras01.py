import tensorflow as tf
print(tf.__version__)   #2.9.3
import numpy as np
print(np.__version__)   #1.21.1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1.데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2.모델구성
model = Sequential()
model.add(Dense(1, input_dim=1)) # Dense(1, input_dim=1) 하나를 넣을거야~ -> 하나를 뱉어~

# 3.컴파일(컴퓨터가 알아쳐먹게 하는 과정), 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100) # 반복하는 과정에서 가중치를 스스로 생성 - 최적의 가중치를 찾아가는 과정

# 4.평가, 예측
result = model.predict(np.array([101]))
print('10의 예측값: ', result)




