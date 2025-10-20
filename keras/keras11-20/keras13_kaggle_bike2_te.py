# https://www.kaggle.com/c/bike-sharing-demand/submissions

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
y =  train_csv[['casual', 'registered']]
def RMSE(a,b) : return np.sqrt(mean_squared_error(a,b))
import random
n = random.randint(1, 10000)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=n)

model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(2, activation='linear'))  

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=32)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)

test2_csv = test_csv.copy()     # 원래 .copy()를 사용해야 함. -> 사본을 만들겠다.
y_submit = model.predict(test_csv)
print('test_csv 타입 :', type(test_csv))    # test_csv 타입 : <class 'pandas.core.frame.DataFrame'> 
                                            # -> pandas 는 딱 두개만 있음 <series(벡터) / dataframe(행렬)> 
print('y_submit 타입 :', type(y_submit))    # y_submit 타입 : <class 'numpy.ndarray'> -> 여기에는 numpy만 있음
                                            # pandas의 데이터는 numpy / pandas에서 헤더와 인덱스를 인식함. numpy로 된 y_submit의 데이터를 넣을 수 있음 

# exit()
test2_csv[['casual','registered']] = y_submit
file = 'new_test_test.csv'
test2_csv.to_csv(path + file, index=False)
# print(test2_csv)
# exit()
print('lotto :', n)
print('file :', file)
print('RMSE :', rmse)
print('R2 :', r2)
