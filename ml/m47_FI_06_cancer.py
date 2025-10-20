from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random

#1. 데이터
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

x = dataset.data    #(569, 30)
y = dataset.target  #(569,)

print(type(x))
print(type(dataset))
exit()

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

seed = 72

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model4 = XGBClassifier(random_state=seed)
model4.fit(x_train, y_train)
print('6')
print(f'# {model4.__class__.__name__}')
print(f'# ACC : {model4.score(x_test, y_test)}') 
print('#', model4.feature_importances_)

print('# 25%지점 :', np.percentile(model4.feature_importances_, 25))
per = np.percentile(model4.feature_importances_, 25)

col_names = []
# 삭제할 컬럼(25% 이하) 찾기
for i, fi in enumerate(model4.feature_importances_) :
    # print(i, fi)
    if fi <= per :
        col_names.append(dataset.feature_names[i])
    else :
        continue

x = pd.DataFrame(x, columns=dataset.feature_names)
x = x.drop(columns=col_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model4.fit(x_train, y_train)
print(f'# ACC2 : {model4.score(x_test, y_test)}')

# XGBClassifier
# ACC : 0.9912280701754386
# [0.03899821 0.02007599 0.00867345 0.04105973 0.00509214 0.00281381
#  0.01790295 0.01979863 0.00167072 0.00761755 0.01876996 0.0048802
#  0.0272005  0.00679698 0.00888047 0.01291873 0.00238045 0.01151786
#  0.00295481 0.00142329 0.13189764 0.01883145 0.20633852 0.01985464
#  0.01064892 0.00659898 0.03191127 0.310359   0.00213313 0.        ]
# 25%지점 : 0.004933184711262584
# ACC2 : 0.9824561403508771