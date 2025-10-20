import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler

#1. 데이터
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, shuffle=True, random_state=304)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

#2. 모델
model = BaggingRegressor(
    DecisionTreeRegressor(),
    n_estimators=100,
    random_state=333,
    # bootstrap=False,
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
results = model.score(x_test, y_test)
print('최종 점수:', results)

# <DecisionTreeRegressor>
# 최종 점수: 0.6361234537246888
# <BaggingRegressor> - bootstrap = True     # default - 샘플 데이터 중복 허용
# 최종 점수: 0.8086336918335213
# <RandomForestRegressor>
# 최종 점수: 0.8095339283632206
# <BaggingRegressor> - bootstrap = False    # 샘플 데이터 중복 불허
# 최종 점수: 0.6570885400256427






