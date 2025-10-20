# 08-1 copy

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])

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
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=3)    
# verbose=0 : 침묵 -> false
#         1 : default -> True
#         2: progress bar 제거 
#         이외 : 에포만 나옴.

loss = model.evaluate(x_test, y_test)
result = model.predict([11])

print('#####################')
print('loss :', loss)
print('예측값 :', result)