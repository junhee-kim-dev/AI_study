import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,  8,  7,  6,  5,  4,  3,  2,  1,  0]])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x = np.transpose(x) #위에 정상이라는 모습으로 수정해줌 (단 너무 큰 데이터면 못함.)

print(x.shape)  #(3, 10) ->(transpose) (10,3)
print(y.shape)  #(10,)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

epochs=100
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=epochs, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
results = model.predict(np.array([[11,2.0,-1]])) #열을 맞춘 값을 요구해야 함 / 행무시! 열우선!
print('################')
print('loss :',loss)
print('[11,2.0,-1]의 예측값 :',results)

# ################
# loss : 1.3670841802343459e-12
# [11,2.0,-1]의 예측값 : [[10.999999]]