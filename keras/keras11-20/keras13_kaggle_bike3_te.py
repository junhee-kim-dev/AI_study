# https://www.kaggle.com/c/bike-sharing-demand/submissions

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

path = ('./_data/kaggle/bike-sharing-demand/')      
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
new_test_csv = pd.read_csv(path + 'new_test_test.csv')
submission_csv = pd.read_csv(path + 'sampleSubmission.csv') #outlier : 이상치
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
def RMSE(a,b) : return np.sqrt(mean_squared_error(a,b))
import random
n = random.randint(1, 10000)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=n)

model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(1, activation='linear'))    # activation='linear' 이게 모델 구성 default

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)

new_test_csv_copy = new_test_csv.copy()
y_submit = model.predict(new_test_csv_copy)
submission_csv['count'] = y_submit
# print(submission_csv)   #(6493,2)
print(new_test_csv_copy)
print(y_submit)
exit()

file = 'submission_0523_2.csv'
submission_csv.to_csv(path + file, index=False)

print('lotto :', n)
print('file :', file)
print('RMSE :', rmse)
print('R2 :', r2)
