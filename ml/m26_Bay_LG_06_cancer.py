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


import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping, log_evaluation
from bayes_opt import BayesianOptimization

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, shuffle=True, random_state=304)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.9, shuffle=True, random_state=333
)

parameters = {
    'learning_rate': (0.005, 0.3),
    'n_estimators': (100, 2000),
    'max_depth': (3, 15),
    'num_leaves': (20, 150),
    'min_child_samples': (5, 100),
    'min_child_weight': (1e-3, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'colsample_bylevel': (0.5, 1.0),
    'reg_alpha': (0.0, 10.0),
    'reg_lambda': (0.0, 10.0),
    'max_bin': (63, 512),
    'min_split_gain': (0, 5)
}

def LGBM_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha, n_estimators, colsample_bylevel, min_split_gain):
    params = {
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),
        'num_leaves': int(round(num_leaves)),
        'min_child_samples' : int(round(min_child_samples)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1), 0),
        'colsample_bytree' : colsample_bytree,
        'max_bin' : int(round(max_bin)),
        'reg_lambda' : max(reg_lambda, 0),          # 디폴트 1 // L2 정규화 // 릿지
        'reg_alpha' : reg_alpha,                    # 디폴트 0 // L1 정규화 // 라쏘
        'n_estimators': int(round(n_estimators)),
        'min_split_gain' : min_split_gain
    }
    
    model = LGBMClassifier(**params, force_col_wise=True, verbosity=-1)
    model.fit(x_train, y_train,
               eval_set=[(x_val, y_val)],
               callbacks=[early_stopping(stopping_rounds=10), log_evaluation(period=0)])
    y_pred = model.predict(pd.DataFrame(x_test))
    result = accuracy_score(y_test, y_pred)

    return result


optimizer = BayesianOptimization(
    f = LGBM_hamsu,
    pbounds=parameters,
    random_state=333
)
optimizer.maximize(init_points=20, n_iter=60)
best_param = optimizer.max
print('추천 조합 :', best_param)

# 추천 조합 : {'target': 0.9649122807017544, 
# 'params': {'colsample_bylevel': 0.7716455435126173, 'colsample_bytree': 0.8644753639495034, 
# 'learning_rate': 0.009980026581737599, 'max_bin': 211.3221225082837, 
# 'max_depth': 7.424661898614874, 'min_child_samples': 9.588848246201557, 
# 'min_child_weight': 1.046197351536536, 'min_split_gain': 0.4871875780818269, 
# 'n_estimators': 566.2662975757532, 'num_leaves': 133.55361607964542, 
# 'reg_alpha': 0.5161163930274903, 'reg_lambda': 0.6048133573318915, 
# 'subsample': 0.6193700568007332}}


