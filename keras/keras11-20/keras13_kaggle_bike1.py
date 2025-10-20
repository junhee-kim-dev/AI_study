# https://www.kaggle.com/c/bike-sharing-demand/submissions

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

#1. 데이터
path = ('./_data/kaggle/bike-sharing-demand/')      
#region(경로 예시)
# path = ('.\_data\kaggle\bike-sharing-demand\')    # \n \a \b 등 예약어를 제외하면 가능
# path = ('.//_data//kaggle//bike-sharing-demand//')# 상대 경로 # !!!!!!제일 많은 실수는 오타와 경로 실수!!!!!!
# path = ('.\\_data\\kaggle\\bike-sharing-demand\\')
# path = 'c: /_data/kaggle/bike-sharing-demand/'
#endregion

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv') #outlier : 이상치
#region(데이터 정보)
# print(train_csv)            #[10886 rows x 12 columns]
# print(train_csv.shape)      #(10886, 12)
# print(test_csv.shape)       #(6493, 9)
# print(submission_csv.shape) #(6493, 2)

# 컬럼 확인
# print(train_csv.columns)
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'],
# print(test_csv.columns)
# Index(['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
#        'atemp', 'humidity', 'windspeed'], dtype='object')
# print(submission_csv.columns)
# Index(['datetime', 'count'], dtype='object')

# 결측치 확인
# print(train_csv.info())
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   datetime    10886 non-null  object
#  1   season      10886 non-null  int64
#  2   holiday     10886 non-null  int64
#  3   workingday  10886 non-null  int64
#  4   weather     10886 non-null  int64
#  5   temp        10886 non-null  float64
#  6   atemp       10886 non-null  float64
#  7   humidity    10886 non-null  int64
#  8   windspeed   10886 non-null  float64
#  9   casual      10886 non-null  int64
#  10  registered  10886 non-null  int64
#  11  count       10886 non-null  int64
# print(train_csv.isnull().sum())           #결측치 없음
# datetime      0
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0
# casual        0
# registered    0
# count         0
# print(test_csv.isna().sum())              #결측치 없음
# datetime      0
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# atemp         0
# humidity      0
# windspeed     0

# print(train_csv.describe())
# print(test_csv.describe())

#endregion

##### x와 y 분리 #####
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
# print(x)        #[10886 rows x 8 columns]
y =  train_csv['count']
# print(y)
# print(y.shape)  #(10886,)
def RMSE(a,b) : return np.sqrt(mean_squared_error(a,b))
import random
n = random.randint(1, 10000)
# 4245

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=n)

model = Sequential()
model.add(Dense(100, input_dim=8, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dense(75, activation='relu'))
model.add(Dense(1, activation='linear'))    # activation='linear' 이게 모델 구성 default

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=32)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)

rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)


y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit
# print(submission_csv)   #(6493,2)

file = 'submission_0522_2.csv'
submission_csv.to_csv(path + file, index=False)

print('lotto :', n)
print('file :', file)
print('RMSE :', rmse)
print('R2 :', r2)

# lotto : 4245
# file : submission_0522_1.csv
# RMSE : 146.88364163117106
# R2 : 0.32755256044332637

# lotto : 4245
# file : submission_0522_2.csv
# RMSE : 145.6944952187336
# R2 : 0.33839653978743933
