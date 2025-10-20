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

# 시드 고정
seed = 123
import random
random.seed(seed)
np.random.seed(seed)

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

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
x_pf = pf.fit_transform(x)
print(x_pf)

x_ori = x.copy()

x_set = [x_ori, x_pf]
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

for id, i in enumerate(x_set):
        x_train, x_test, y_train, y_test = train_test_split(
            i, y, random_state=123, train_size=0.8,
        )

        model = XGBRegressor(random_state=123)

        model.fit(x_train, y_train)
        results = model.predict(x_test)

        r2 = r2_score(y_test, results)

        print(f"{id} 의 R2는 {r2:.4f}입니다!")