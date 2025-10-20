from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint

path ='./Study25/_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv')

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
print(x.shape)  #(1459, 9)
print(y.shape)  #(1459,)


import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

seed = 72

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed
)

model4 = XGBRegressor(random_state=seed)
model4.fit(x_train, y_train)
print('4')
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
        col_names.append(x.columns[i])
    else :
        continue

x = pd.DataFrame(x, columns=x.columns)
x = x.drop(columns=col_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed
)

model4.fit(x_train, y_train)
print(f'# ACC2 : {model4.score(x_test, y_test)}')

# 4
# XGBRegressor
# ACC : 0.7600917094435791
# [0.34762222 0.11409687 0.3687732  0.0226927  0.03534447 0.02287514
#  0.03140607 0.02996508 0.02722422]
# 25%지점 : 0.027224218472838402
# ACC2 : 0.7463736292278567