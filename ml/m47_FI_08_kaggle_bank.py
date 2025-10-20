# https://www.kaggle.com/competitions/playground-series-s4e1/submissions

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import f1_score
from keras.layers import BatchNormalization
from keras.layers import Dropout
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = './Study25/_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

train_csv[['Tenure', 'Balance']] = train_csv[['Tenure', 'Balance']].replace(0, np.nan)
train_csv = train_csv.fillna(train_csv.mean())

test_csv[['Tenure', 'Balance']] = test_csv[['Tenure', 'Balance']].replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

oe = OrdinalEncoder()       # 이렇게 정의 하는 것을 인스턴스화 한다고 함
oe.fit(train_csv[['Geography', 'Gender']])
train_csv[['Geography', 'Gender']] = oe.transform(train_csv[['Geography', 'Gender']])
test_csv[['Geography', 'Gender']] = oe.transform(test_csv[['Geography', 'Gender']])

train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']



import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

seed = 72

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model4 = XGBClassifier(random_state=seed)
model4.fit(x_train, y_train)
print('8')
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
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model4.fit(x_train, y_train)
print(f'# ACC2 : {model4.score(x_test, y_test)}')

# 8
# XGBClassifier
# ACC : 0.8637258763292635
# [0.00965877 0.03852049 0.06342223 0.12794696 0.00856875 0.02582036
#  0.5086978  0.0119453  0.1960336  0.00938571]
# 25%지점 : 0.010230403393507004
# ACC2 : 0.8643015118005272