import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout, BatchNormalization, Conv1D

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10]) # 시계열이 될 수도 있고 아닐 수도 있다. -> 데이터에 종속되지 마라

# 시계열 데이터라면 time_step에 따라 잘라내야함
x = np.array([[1,2,3],  # 다음은 4야~
              [2,3,4],  # 다음은 5야~
              [3,4,5],  # 다음은 6이야~
              [4,5,6],  # 다음은 7이야~
              [5,6,7],  # 다음은 8이야~
              [6,7,8],  # 다음은 9야~
              [7,8,9],  # 다음은 10이야~ -> 8,9,10 다음은 뭐게~? -> 이런 게 RNN의 작동 목적
             ])
y = np.array([ 4, 5, 6, 7, 8, 9,10])    # -> datasets 을 x, y로 이렇게 잘라야 함

x = x.reshape(x.shape[0], x.shape[1], 1)


# 2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=100, input_shape=(3,1), activation='relu'))    # 시계열 데이터의 아웃풋은 2차원이다. 시계열 데이터가 섞이면서 순서가 사라짐
# model.add(LSTM(10, input_shape=(3,1),))
# model.add(GRU(10, input_shape=(3,1), activation='relu'))
model.add(Conv1D(filters=10, kernel_size=2, input_shape=(3,1)))
model.add(Dense(128,))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam',)
model.fit(x, y, epochs=3000,)

#4. 평가 예측
loss = model.evaluate(x,y)
print('loss:', loss)

x_pred = np.array([8,9,10]).reshape(1,3,1)
results = model.predict(x_pred)
print('[8,9,10]의 결과 :', results)










