from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 에포는 100으로 고정
# loss 기준 0.32 미만로 만들것

#2. 모델구성
model = Sequential()
model.add(Dense(100,input_dim=1)) # 어차피 상위 레이어의 아웃풋은 하위 레이어의 인풋이기 때문에 아웃풋만 표기
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

epochs=100 # 고정
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('#######################')
print('epochs : ',epochs)
print('loss : ',loss)
results = model.predict(np.array([7]))
print('예측값 : ',results)

# #######################
# epochs :  100
# loss :  0.3246256411075592