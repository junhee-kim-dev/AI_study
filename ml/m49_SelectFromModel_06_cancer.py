from math import gamma
import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

#1. 데이터
dataset = load_breast_cancer()

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
        eval_metric='logloss',
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
    # print(select_x_train.shape)
    
    select_model = XGBClassifier(
        gamma        = 0,
        subsample    = 0.4,
        reg_alpha    = 0,
        reg_lambda   = 1,
        max_depth    = 6,
        n_estimators = 10000,
        eval_metric='logloss',
        random_state = seed,
        early_stopping_rounds = 20,
        )
    
    select_model.fit(select_x_train, y_train, eval_set =[(select_x_test, y_test)], verbose=0)
    
    select_y_pred = select_model.predict(select_x_test)
    # print(select_y_pred)
    acc = accuracy_score(y_test, select_y_pred)
    print(f'Treshold:{i:.4f}, Columns:{select_x_train.shape[1]}, ACC:{acc*100:.4f}%')
    
    
# Treshold:0.0024, Columns:30, ACC:99.1228%
# Treshold:0.0032, Columns:29, ACC:99.1228%
# Treshold:0.0034, Columns:28, ACC:100.0000%
# Treshold:0.0044, Columns:27, ACC:100.0000%
# Treshold:0.0050, Columns:26, ACC:100.0000%
# Treshold:0.0056, Columns:25, ACC:100.0000%
# Treshold:0.0061, Columns:24, ACC:99.1228%
# Treshold:0.0067, Columns:23, ACC:99.1228%
# Treshold:0.0077, Columns:22, ACC:99.1228%
# Treshold:0.0079, Columns:21, ACC:99.1228%
# Treshold:0.0084, Columns:20, ACC:99.1228%
# Treshold:0.0090, Columns:19, ACC:100.0000%
# Treshold:0.0095, Columns:18, ACC:99.1228%
# Treshold:0.0100, Columns:17, ACC:100.0000%
# Treshold:0.0121, Columns:16, ACC:99.1228%
# Treshold:0.0121, Columns:15, ACC:99.1228%
# Treshold:0.0127, Columns:14, ACC:100.0000%
# Treshold:0.0131, Columns:13, ACC:99.1228%
# Treshold:0.0132, Columns:12, ACC:98.2456%
# Treshold:0.0146, Columns:11, ACC:98.2456%
# Treshold:0.0155, Columns:10, ACC:98.2456%
# Treshold:0.0166, Columns:9, ACC:99.1228%
# Treshold:0.0174, Columns:8, ACC:99.1228%
# Treshold:0.0190, Columns:7, ACC:96.4912%
# Treshold:0.0209, Columns:6, ACC:96.4912%
# Treshold:0.0378, Columns:5, ACC:96.4912%
# Treshold:0.0520, Columns:4, ACC:96.4912%
# Treshold:0.1188, Columns:3, ACC:96.4912%
# Treshold:0.2000, Columns:2, ACC:96.4912%
# Treshold:0.3348, Columns:1, ACC:92.9825%