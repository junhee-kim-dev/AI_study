from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array(range(1,17))
y = np.array(range(1,17))

x_train = x[:10]
y_train = y[:10]
x_val = x[10:13]
y_val = y[10:13]
x_test = x[13:]
y_test = y[13:]

print(x_train)  #[ 1  2  3  4  5  6  7  8  9 10]
print(x_val)    #[11 12 13]
print(x_test)   #[14 15 16]
print(y_train)  #[ 1  2  3  4  5  6  7  8  9 10]
print(y_val)    #[11 12 13]
print(y_test)   #[14 15 16]

model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_data=(x_val, y_val))

loss = model.evaluate(x_test, y_test)
results = model.predict([17])
print('loss :',loss)
print('예측값 :', results)