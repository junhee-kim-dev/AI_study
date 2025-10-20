from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

#1. 데이터
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target
# print(x.shape)  #(506.13)
# print(y.shape)  #(506,)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.2, shuffle=True, random_state=142
)

model = Sequential()
model.add(Dense(50, input_dim = 13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(
    x_train, y_train, 
    epochs=100, batch_size=1, 
    verbose=1, validation_split=0.3
)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = np.sqrt(loss)
r2 = r2_score(y_test, results)

print('###################')
print('RMSE :', rmse)   
print('R2 :', r2)

#전
# ###################
# RMSE : 4.827135674076012
# R2 : 0.756898103912625

#후
# ###################
# RMSE : 5.60012539314481
# R2 : 0.6256338850238707