from math import gamma
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

#1. 데이터
dataset = load_iris()

x = dataset.data
y = dataset.target

le = LabelEncoder()
y = le.fit_transform(y)

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import EarlyStopping

seed = 72

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)
model4 = XGBClassifier(
        gamma        = 0,
        subsample    = 0.4,
        reg_alpha    = 0,
        reg_lambda   = 1,
        max_depth    = 6,
        n_estimators = 10000,
        eval_metric='mlogloss',
        random_state = seed,
        early_stopping_rounds = 20,
)

model4.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose = 0)
# print('########### ')
# print(f'#   기존 R2 : {model4.score(x_test, y_test)}') 

thresholds = np.sort(model4.feature_importances_)

from sklearn.feature_selection import SelectFromModel
for i in thresholds :
    selection = SelectFromModel(model4, threshold=i, prefit=True)
    # threshold 가 i값 이상인 것을 모두 훈련시킨다.
    # prefit = False : 모델이 아직 학습되지 않았을 때, fit을 호출해서 훈련한다.(default)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape)
    
    select_model = XGBClassifier(
        gamma        = 0,
        subsample    = 0.4,
        reg_alpha    = 0,
        reg_lambda   = 1,
        max_depth    = 6,
        n_estimators = 10000,
        eval_metric='mlogloss',
        random_state = seed,
        early_stopping_rounds = 20,
        )
    
    select_model.fit(select_x_train, y_train, eval_set =[(select_x_test, y_test)], verbose=0)
    
    select_y_pred = select_model.predict(select_x_test)
    # print(select_y_pred)
    acc = accuracy_score(y_test, select_y_pred)
    print(acc)
    
########### 
# (120, 4)
# 1.0
# (120, 3)
# 0.9666666666666667
# (120, 2)
# 1.0
# (120, 1)
# 0.9666666666666667