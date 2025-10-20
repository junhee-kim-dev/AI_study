import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5 ],
              [6,7,8,9,10]]) #x=np.array([[1,6],[2,7],[3,8],[4,9],[5,10]]) 이게 정상
y = np.array([1,2,3,4,5])
x = np.transpose(x) #위에 정상이라는 모습으로 수정해줌 (단 너무 큰 데이터면 못함.)

print(x.shape)  #(2, 5) ->(transpose) (5,2)
print(y.shape)  #(5,)

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=2))
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
results = model.predict([[6,11]]) #(1,2) 열을 맞춘 값을 요구해야 함 / 행무시! 열우선!
print('################')
print('loss :',loss)
print('[6,11]의 예측값 :',results)

################
# loss : 8.981260288427884e-13
# [6,11]의 예측값 : [[6.]]

# 