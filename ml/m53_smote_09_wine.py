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

################## SMOTE 적용 ####################
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=seed,
              k_neighbors=5,           #default
            #   sampling_strategy='auto' #default
            #   sampling_strategy=0.75   # 최대값의 75% 수준
              sampling_strategy={0:5000, 1:5000, 2:5000},
            #   n_jobs=-1   # 0.13버전에서 삭제
              )

x_train, y_train = smote.fit_resample(x_train, y_train)
print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2]), array([5000, 5000, 5000]))
print(x_train.shape)    #(15000, 13)
exit()

# 2. 모델
model = Sequential()
model.add(Dense(10,input_shape=(13,)))
model.add(Dense(3,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

results =model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc  :', results[1])

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
# print(y_pred.shape)       (35, 1)
# print(y_pred)   [1 0 0 0 0 1 0 0 0 1 0 0 1 0 1 1 0 0 1 1 0 1 2 0 0 1 1 1 1 0 1 2 0 0 0]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print(acc)
print(f1)

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



# 데이터가 엄청 많은데 불균형 하다면 비율대로 줄여서 계산하면 그 비율만큼 smote 연산량이 줄어듦
# if 
# {0: 100000, 1:30000, 2:10000} 칼럼 10개짜리를 smote 시키면 계산을 300억 번 계산해야하는데
# 각 갯수를 1:10 으로 나눠서 계산하면 30억번만 계산하면 됨
# 각 갯수를 1:100으로 나누면 3억번