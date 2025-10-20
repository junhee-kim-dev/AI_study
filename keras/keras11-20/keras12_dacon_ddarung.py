#region(from, import)

# https://dacon.io/competitions/official/235576/overview/description
import numpy as np      # print(np.__version__)   # 1.23.0
import pandas as pd     # print(pd.__version__)   # 2.2.3
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
#endregion

#region
import random

n = random.randint(1, 100000)
e = random.randint(200, 1000)
b = random.randint(16, 65) 
l1 = random.randint(50, 200)
l2 = random.randint(200, 300)
l3 = random.randint(100, 200)
l4 = random.randint(50, 100)
#endregion

#region(내용물)
#1. 데이터
#region
path = './_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv  = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv', index_col=0)

#region(데이터 정보)
# print(train_csv)
# (1459,10) 열을 9/1로 나눠야하는데...
# print(test_csv)
# (715,9)
# print(submission_csv) 
# (715, 1)


# print(train_csv.shape)
# (1459, 10)
# print(test_csv.shape)
# (715, 9)
# print(submission_csv.shape)
# (715, 1)
# print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
# print(train_csv.info())
#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    1459 non-null   int64    # 결측치 확인 -> data의 총 개수 보다 적은 부분
#  1   hour_bef_temperature    1457 non-null   float64   
#  2   hour_bef_precipitation  1457 non-null   float64   
#  3   hour_bef_windspeed      1450 non-null   float64   
#  4   hour_bef_humidity       1457 non-null   float64   
#  5   hour_bef_visibility     1457 non-null   float64   
#  6   hour_bef_ozone          1383 non-null   float64   
#  7   hour_bef_pm10           1369 non-null   float64   
#  8   hour_bef_pm2.5          1342 non-null   float64   
#  9   count                   1459 non-null   float64   
# print(train_csv.describe())
# exit()
#endregion

# print(train_csv.isnull().sum()) #결측치의 개수 출력 print(train_csv.isna().sum()) #결측치의 개수 출력

#######################   결측치 처리 1. 삭제     ##############################

# train_csv=train_csv.dropna()
# print(train_csv)    #[1328 rows x 10 columns]
# print(train_csv.isna().sum())
# print(train_csv.info())

#region(dropna() 보정 후 데이터 정보)

# hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0
# count                     0
# dtype: int64
# <class 'pandas.core.frame.DataFrame'>
# Index: 1328 entries, 3 to 2179
# Data columns (total 10 columns):
#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    1328 non-null   int64
#  1   hour_bef_temperature    1328 non-null   float64
#  2   hour_bef_precipitation  1328 non-null   float64
#  3   hour_bef_windspeed      1328 non-null   float64
#  4   hour_bef_humidity       1328 non-null   float64
#  5   hour_bef_visibility     1328 non-null   float64
#  6   hour_bef_ozone          1328 non-null   float64
#  7   hour_bef_pm10           1328 non-null   float64
#  8   hour_bef_pm2.5          1328 non-null   float64
#  9   count                   1328 non-null   float64
# dtypes: float64(9), int64(1)
#endregion

#######################결측치 처리 2. 평균값 넣기 ##############################

train_csv = train_csv.fillna(train_csv.mean())
# print(train_csv)    #[1459 rows x 10 columns]
# print(train_csv.isna().sum())
# print(train_csv.info())

#region(fillna() 보정 후 데이터 정보)

# hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0
# count                     0
# dtype: int64
# <class 'pandas.core.frame.DataFrame'>
# Index: 1459 entries, 3 to 2179
# Data columns (total 10 columns):
#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1459 non-null   float64
#  2   hour_bef_precipitation  1459 non-null   float64
#  3   hour_bef_windspeed      1459 non-null   float64
#  4   hour_bef_humidity       1459 non-null   float64
#  5   hour_bef_visibility     1459 non-null   float64
#  6   hour_bef_ozone          1459 non-null   float64
#  7   hour_bef_pm10           1459 non-null   float64
#  8   hour_bef_pm2.5          1459 non-null   float64
#  9   count                   1459 non-null   float64
# dtypes: float64(9), int64(1)
#endregion

#####################아이구 테스트도 결측이 있대요.##############################
# !!!!테스트의 결측치는 평균값 넣기로 해야함!!!!
test_csv = test_csv.fillna(test_csv.mean())
# print(test_csv)     #[715 rows x 9 columns]
# print(test_csv.isna().sum())
# print(test_csv.info())

#region(테스트 결측치 확인 및 평균값 넣기)
# hour                        0
# hour_bef_temperature        2
# hour_bef_precipitation      2
# hour_bef_windspeed          9
# hour_bef_humidity           2
# hour_bef_visibility         2
# hour_bef_ozone             76
# hour_bef_pm10              90
# hour_bef_pm2.5            117
# count                       0
# dtype: int64
# hour                      0
# hour_bef_temperature      0
# hour_bef_precipitation    0
# hour_bef_windspeed        0
# hour_bef_humidity         0
# hour_bef_visibility       0
# hour_bef_ozone            0
# hour_bef_pm10             0
# hour_bef_pm2.5            0
# dtype: int64
# <class 'pandas.core.frame.DataFrame'>
# Index: 715 entries, 0 to 2177
# Data columns (total 9 columns):
#  #   Column                  Non-Null Count  Dtype
# ---  ------                  --------------  -----
#  0   hour                    715 non-null    int64
#  1   hour_bef_temperature    715 non-null    float64
#  2   hour_bef_precipitation  715 non-null    float64
#  3   hour_bef_windspeed      715 non-null    float64
#  4   hour_bef_humidity       715 non-null    float64
#  5   hour_bef_visibility     715 non-null    float64
#  6   hour_bef_ozone          715 non-null    float64
#  7   hour_bef_pm10           715 non-null    float64
#  8   hour_bef_pm2.5          715 non-null    float64
# dtypes: float64(8), int64(1)
#endregion

c = 'count'
x = train_csv.drop([c], axis=1)   # 행 또는 열 삭제 # count라는 axis=1 -> 열 삭제, 참고로 행은 axis=0 -> 행 삭제
# print(x)            #[1459 rows x 9 columns]
y = train_csv[c]
# print(y.shape)      #(1459,)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.75, test_size=0.25, shuffle=True, random_state=n
    )
#endregion

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=9, activation='relu'))
model.add(Dense(l1, activation='relu'))
model.add(Dense(l2, activation='relu'))
model.add(Dense(l3, activation='relu'))
model.add(Dense(l4, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=e, batch_size=b)

#4. 평가, 예측
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

#region(결과)

# x, y, train_size=0.7, test_size=0.3, shuffle=True, random_state=9
# epochs=100, batch_size=32
#######################   결측치 처리 1. 삭제     ##############################
# 1회차
# loss : 2860.136962890625
# RMSE : 53.48024738336847
# R2_1 = 0.5629606659913835

# 2회차
# loss : 3330.319580078125
# RMSE : 57.70892390267423
# R2_2 = 0.491114984558888

# 3회차
# loss : 3322.3076171875
# RMSE : 57.63946182414288
# R2_3 = 0.4923392990965939

# 4회차
# loss : 2875.540771484375
# RMSE : 53.62406834315445
# R2_4 = 0.5606069022727125

# 5회차
# loss : 3034.362060546875
# RMSE : 55.085043120014035
# R2_5 = 0.5363384335244752

# print((R2_1+R2_2+R2_3+R2_4+R2_5)/5) 
# 0.5286720570888106

#######################결측치 처리 2. 평균값 넣기 ##############################
# 1회차
# loss : 2623.870361328125
# RMSE : 51.22373231053984
# R2_1 = 0.5601837857676275

# 2회차
# loss : 3205.045166015625
# RMSE : 56.613120536872515
# R2_2 = 0.4627666241811279

# 3회차
# loss : 2605.340576171875
# RMSE : 51.042534524649405
# R2_3 = 0.563289876134415

# 4회차
# loss : 2616.68603515625
# RMSE : 51.15355676278376
# R2_4 = 0.561388040146211

# 5회차
# loss : 2709.1767578125
# RMSE : 52.04975360552372
# R2_5 = 0.5458846800286541

# print((R2_1+R2_2+R2_3+R2_4+R2_5)/5) 
# 0.5387026012516071

#endregion(결과)

############ submission.csv에 test_csv의 예측값 넣기#############
y_submit = model.predict(test_csv) # train 데이터의 shape과 동일한 칼럼을 확인하고 넣어야 함
# print(x_train.shape)    # (1021, 9)
# print(y_submit.shape)   # (715, 1)

####### submission.csv 파일 만들기 // count 컬럼값만 넣어주기#######
# print(submission_csv)                 # 전부 NaN
submission_csv['count'] = y_submit
# print(submission_csv)                 # count 열에 다 들어감
#endregion

submission_csv.to_csv(path + 'submission_0522_2.csv')  # csv파일 만들기

#region(결과값 0521)
# epochs=300, batch_size=32, random_state=123
###############
# submission_0521_1.csv
# RMSE : 30.429473980305207
# R2 : 0.845709491395605
###############
# submission_0521_2.csv
# RMSE : 27.84453591100831
# R2 : 0.8708095864771599
###############
# submission_0521_3.csv
# RMSE : 39.962595700580515
# R2 : 0.7338921419264872
###############
#endregion

#region(결과값 0522)
###############
# submission_0522_1.csv
# random : 2015
# epochs : 550
# batch : 32
# loss : 2262.05712890625
# RMSE : 47.561089310937525
# R2 : 0.6737186603871579
###############
# submission_0522_2.csv
# RMSE : 27.84453591100831
# R2 : 0.8708095864771599
###############
# submission_0522_3.csv
# RMSE : 39.962595700580515
# R2 : 0.7338921419264872
###############
#endregion