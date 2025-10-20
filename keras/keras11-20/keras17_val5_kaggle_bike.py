# https://www.kaggle.com/c/bike-sharing-demand/submissions

# 인증서 오류 일때
# import ssl
# # import certifi
# # import urllib.request
# ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

path = ('./_data/kaggle/bike-sharing-demand/')    
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv')

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y =  train_csv['count']
def RMSE(a,b) : return np.sqrt(mean_squared_error(a,b))
import random
n = 145 #random.randint(1, 10000)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=n)

model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(1, activation='linear')) 

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=32, verbose=2, validation_split=0.2)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)


# y_submit = model.predict(test_csv)
# submission_csv['count'] = y_submit
# # print(submission_csv)   #(6493,2)

# file = 'submission_0522_2.csv'
# submission_csv.to_csv(path + file, index=False)

print('lotto :', n)
# print('file :', file)
print('RMSE :', rmse)
print('R2 :', r2)

#전
# lotto : 4245
# RMSE : 145.6944952187336
# R2 : 0.33839653978743933

#후
# lotto : 145
# RMSE : 154.44949539676242
# R2 : 0.27248546347662284
