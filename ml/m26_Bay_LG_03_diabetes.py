from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from keras.callbacks import EarlyStopping

dataset = load_diabetes()
x = dataset.data
y = dataset.target


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
    
    model = LGBMRegressor(**params, force_col_wise=True, verbosity=-1)
    model.fit(x_train, y_train,
               eval_set=[(x_val, y_val)],
               callbacks=[early_stopping(stopping_rounds=10), log_evaluation(period=0)])
    y_pred = model.predict(pd.DataFrame(x_test))
    result = r2_score(y_test, y_pred)

    return result


optimizer = BayesianOptimization(
    f = LGBM_hamsu,
    pbounds=parameters,
    random_state=333
)
optimizer.maximize(init_points=20, n_iter=60)
best_param = optimizer.max
print('추천 조합 :', best_param)

#       best_params : {'min_child_weight': 10, 'learning_rate': 0.1}
#        best_score : 0.3767878412254125
#  model_best_score : 0.40681174031628675
#    accuracy_score : 0.40681174031628675
#      running_time : 3.932 sec

#       best_params : {'learning_rate': 0.1, 'min_child_weight': 10}
#        best_score : 0.38311797608602555
#  model_best_score : 0.40681174031628675
#    accuracy_score : 0.40681174031628675
#      running_time : 3.145 sec

# 추천 조합 : {'target': 0.5241296882850618, 
# 'params': {'colsample_bylevel': 0.7382891675551423, 
# 'colsample_bytree': 0.6214860477793445, 'learning_rate': 0.24525548896705388, 
# 'max_bin': 373.8189248775181, 'max_depth': 11.608351262356368, 
# 'min_child_samples': 8.018525539197592, 'min_child_weight': 4.416224877659587, 
# 'min_split_gain': 1.1925655184426032, 'n_estimators': 1750.824330621237, 
# 'num_leaves': 99.37351093134717, 'reg_alpha': 6.474728883862771, 
# 'reg_lambda': 4.1704117618150125, 'subsample': 0.5510418154823631}}