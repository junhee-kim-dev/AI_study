# import ssl

# # SSL 인증서 문제 해결
# ssl._create_default_https_context = ssl._create_unverified_context

from sklearn.datasets import fetch_covtype
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import numpy as np
import pandas as pd

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import LabelEncoder
la = LabelEncoder()
y = la.fit_transform(y)

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

seed = 72

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model4 = XGBClassifier(random_state=seed)
model4.fit(x_train, y_train)
print('10')
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
        col_names.append(datasets.feature_names[i])
    else :
        continue

x = pd.DataFrame(x, columns=datasets.feature_names)
x = x.drop(columns=col_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model4.fit(x_train, y_train)
print(f'# ACC2 : {model4.score(x_test, y_test)}')

# XGBClassifier
# ACC : 0.869392356479609
# [0.09185332 0.00714012 0.00427741 0.01293598 0.00703897 0.01299726
#  0.00823285 0.01116424 0.00492933 0.01234658 0.05916592 0.03107468
#  0.02950124 0.02342194 0.0038407  0.04932087 0.01955242 0.04579308
#  0.00510609 0.00567793 0.00163142 0.00774856 0.01189664 0.01443456
#  0.01275748 0.04035191 0.01162157 0.0033533  0.         0.00661046
#  0.01074708 0.00642328 0.00621943 0.01274691 0.01897344 0.05548857
#  0.02759144 0.01556923 0.00769642 0.00574218 0.01854873 0.00274128
#  0.02663382 0.02331711 0.02205056 0.04403189 0.01859603 0.00505556
#  0.01744643 0.0032799  0.00911526 0.03729373 0.03540351 0.01351136]
# 25%지점 : 0.006470071733929217
# ACC2 : 0.8750204383707821