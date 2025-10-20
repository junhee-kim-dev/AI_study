# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
import datetime
import time

path = './Study25/_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, shuffle=True, random_state=304)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBClassifier

print('시작')
#2. 모델
model = BaggingClassifier(
    XGBClassifier(max_depth=8),
    n_estimators=100,
    random_state=333,
    verbose=1,
    n_jobs=20
)

#3. 훈련
model.fit(x_train, y_train)
print('끝')
#4. 평가
results = model.score(x_test, y_test)
print('최종 점수:', results)

# 최종 점수: 0.91315

