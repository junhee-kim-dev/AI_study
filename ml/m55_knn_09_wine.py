# smote는 보간법

import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

# 시드 고정
seed = 123
import random
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)                     # (178, 13) (178,)
# print(np.unique(y, return_counts=True))     # (array([0, 1, 2]), array([59, 71, 48]))
# print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
# dtype: int64

# print(y)

### 데이터 삭제
y = y[:-40]
x = x[:-40]
# print(y.shape)  (138,)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2]
# print(x.shape)  (138, 13)
# print(np.unique(y, return_counts=True))     #(array([0, 1, 2]), array([59, 71,  8]))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=seed, train_size=0.75, stratify=y
)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('ACC :', acc)
f1 = f1_score(y_test, y_pred, average='macro')
print('F1  :', f1)

# 원데이터
# 0.8888888888888888
# 0.8872800402212166

# 40개 삭제
# 0.8
# 0.5649509803921569

# smote
# default
# 0.8571428571428571
# 0.7094854070660522

# smote
# sampling_strategy={0:50, 2:33}
# 0.9428571428571428
# 0.8130568356374809

# smote
# sampling_strategy={0:5000, 1:5000, 2:5000},
# 1.0
# 1.0

# ACC : 0.8285714285714286
# F1  : 0.5775401069518716


# 데이터가 엄청 많은데 불균형 하다면 비율대로 줄여서 계산하면 그 비율만큼 smote 연산량이 줄어듦
# if 
# {0: 100000, 1:30000, 2:10000} 칼럼 10개짜리를 smote 시키면 계산을 300억 번 계산해야하는데
# 각 갯수를 1:10 으로 나눠서 계산하면 30억번만 계산하면 됨
# 각 갯수를 1:100으로 나누면 3억번