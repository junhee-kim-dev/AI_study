# 08-1 카피
# validation 검증/확인

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x_train = np.array([1,2,3,4,5,6])
y_train = np.array([1,2,3,4,5,6])

x_val = np.array([7,8])
y_val = np.array([7,8])

x_test  = np.array([9,10])
y_test  = np.array([9,10])

model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          verbose=2, validation_data=(x_val, y_val),
          )

loss = model.evaluate(x_test, y_test)
result = model.predict([11])

print('#####################')
print('loss :', loss)
print('예측값 :', result)