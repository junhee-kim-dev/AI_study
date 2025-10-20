# 14 copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import time     # 시간에 대한 모듈 import

x_train = np.array(range(100))
y_train = np.array(range(100))

x_test  = np.array([8,9,10])
y_test  = np.array([8,9,10])

model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
start_time = time.time()    #현재 시간을 반환, 시작시간.
# print(start_time)
model.fit(x_train, y_train, epochs=10000, batch_size=128, verbose=1)
end_time = time.time()
# print(end_time)
print('걸린 시간 :', end_time - start_time, '초')

exit()

#1. 1000epochs 에서 0,1,2,3의 시간을 적는다.
# verbosa=0 걸린 시간 : 41.946940898895264 초
# verbosa=1 걸린 시간 : 56.17138743400574 초
# verbosa=2 걸린 시간 : 46.43154287338257 초
# verbosa=3 걸린 시간 : 42.58944368362427 초

#2. verbose = 1일 떄 batch 1, 32, 128 일 떄 시간
# batch_size=  1 걸린 시간 : 53.611316442489624 초
# batch_size= 32 걸린 시간 : 3.523615837097168 초
# batch_size=128 걸린 시간 : 1.7077016830444336 초
