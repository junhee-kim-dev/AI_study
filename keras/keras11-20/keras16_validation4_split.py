# 16-3 카피

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

x = np.array(range(1,170))
y = np.array(range(1,170))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.85, 
    shuffle=True, random_state=15
)

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
    validation_split=0.2,
    # train_test_split으로 쓰지 말고 validation_split으로 쓰는 게 이득
)

loss = model.evaluate(x_test, y_test)
results = model.predict([17])
print('loss :',loss)
print('예측값 :', results)