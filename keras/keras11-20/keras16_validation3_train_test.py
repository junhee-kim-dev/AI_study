from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

x = np.array(range(1,17))
y = np.array(range(1,17))

# x_train = x[:10]
# y_train = y[:10]
# x_val = x[10:13]
# y_val = y[10:13]
# x_test = x[13:]
# y_test = y[13:]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, 
    shuffle=True, random_state=15
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.8, 
    shuffle=True, random_state=14
)

# print(x_train)  #[ 1  4 13 16 14  9  6  2 15 10]
# print(x_val)    #[ 7  3 12]
# print(x_test)   #[ 8 11  5]
# print(y_train)  #[ 1  4 13 16 14  9  6  2 15 10]
# print(y_val)    #[ 7  3 12]
# print(y_test)   #[ 8 11  5]

model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.fit(
    x_train, y_train, epochs=100, 
    batch_size=32, verbose=2, 
    validation_data=(x_val, y_val)
)

loss = model.evaluate(x_test, y_test)
results = model.predict([17])
print('loss :',loss)
print('예측값 :', results)