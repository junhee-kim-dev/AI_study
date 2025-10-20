# https://dacon.io/competitions/official/235576/overview/description
import numpy as np      # print(np.__version__)   # 1.23.0
import pandas as pd     # print(pd.__version__)   # 2.2.3
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import random

n  = 2015#random.randint(1, 100000)
e  = 550#random.randint(200, 1000)
b  = 32#random.randint(16, 65) 
l1 = 100#random.randint(50, 200)
l2 = 100#random.randint(200, 300)
l3 = 100#random.randint(100, 200)
l4 = 100#random.randint(50, 100)

path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv  = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())
c = 'count'
x = train_csv.drop([c], axis=1) 
y = train_csv[c]
x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.75, test_size=0.25, shuffle=True, random_state=n
    )

model = Sequential()
model.add(Dense(100, input_dim=9, activation='relu'))
model.add(Dense(l1, activation='relu'))
model.add(Dense(l2, activation='relu'))
model.add(Dense(l3, activation='relu'))
model.add(Dense(l4, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=e, batch_size=b, verbose=1, validation_split=0.2)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
def RMSE(a, b):
    return np.sqrt(mean_squared_error(a, b))
rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)

print('random :',n)
print('epochs :',e)
print('batch :', b)
print('loss :',loss)
print('RMSE :',rmse)
print('R2 :', r2)

#전
# random : 2015
# epochs : 550
# batch : 32
# loss : 2262.05712890625
# RMSE : 47.561089310937525
# R2 : 0.6737186603871579

#후
# random : 2015
# epochs : 550
# batch : 32
# loss : 3059.0615234375
# RMSE : 55.30878152544156
# R2 : 0.5587580118758163