from math import gamma
import numpy as np

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
dataset = load_diabetes()

x = dataset.data
y = dataset.target

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import EarlyStopping

seed = 72

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed
)
model4 = XGBRegressor(
        gamma        = 0,
        subsample    = 0.4,
        reg_alpha    = 0,
        reg_lambda   = 1,
        max_depth    = 6,
        n_estimators = 10000,
        random_state = seed,
        early_stopping_rounds = 20,
)

model4.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose = 0)
# print('########### ')
print(f'#   기존 R2 : {model4.score(x_test, y_test)}') 

thresholds = np.sort(model4.feature_importances_)

from sklearn.feature_selection import SelectFromModel
for i in thresholds :
    selection = SelectFromModel(model4, threshold=i, prefit=True)
    # threshold 가 i값 이상인 것을 모두 훈련시킨다.
    # prefit = False : 모델이 아직 학습되지 않았을 때, fit을 호출해서 훈련한다.(default)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape)
    
    select_model = XGBRegressor(
        gamma        = 0,
        subsample    = 0.4,
        reg_alpha    = 0,
        reg_lambda   = 1,
        max_depth    = 6,
        n_estimators = 10000,
        random_state = seed,
        early_stopping_rounds = 20,
        )
    
    select_model.fit(select_x_train, y_train, eval_set =[(select_x_test, y_test)], verbose=0)
    
    select_y_pred = select_model.predict(select_x_test)
    # print(select_y_pred)
    acc = r2_score(y_test, select_y_pred)
    print(f'Treshold:{i:.4f}, Columns:{select_x_train.shape[1]}, ACC:{acc*100:.4f}%, ')
    
# Treshold:0.0406, Columns:10, ACC:45.8080%
# Treshold:0.0494, Columns:9, ACC:46.1292%
# Treshold:0.0749, Columns:8, ACC:45.7658%
# Treshold:0.0847, Columns:7, ACC:53.2798%
# Treshold:0.0881, Columns:6, ACC:58.5780%
# Treshold:0.1102, Columns:5, ACC:43.4557%
# Treshold:0.1132, Columns:4, ACC:43.6005%
# Treshold:0.1282, Columns:3, ACC:43.2781%
# Treshold:0.1298, Columns:2, ACC:43.8340%
# Treshold:0.1809, Columns:1, ACC:23.3566%