import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import fetch_california_housing

x, y = fetch_california_housing(return_X_y=True)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, shuffle=True, random_state=304)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

parameters ={
     'n_estimators' : (100,500),
     'learning_rate' : (0.001, 0.1),
     'max_depth' : (3, 10),
     'gamma' : (0,5),
     'min_child_weight' : (1,50),
     'subsample' : (0.5,1),
     'colsample_bytree' : (0.5,1),
     'colsample_bylevel' : (0.5,1),
     'max_bin' : (9,500),
     'reg_lambda' : (0,100),
     'reg_alpha' : (0,10)
     } 


def xgb_hamsu(learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha, n_estimators, gamma, colsample_bylevel):
    params = {
        'learning_rate' : learning_rate,
        'max_depth' : int(round(max_depth)),
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1), 0),
        'colsample_bytree' : colsample_bytree,
        'max_bin' : int(round(max_bin)),
        'reg_lambda' : max(reg_lambda, 0),          # 디폴트 1 // L2 정규화 // 릿지
        'reg_alpha' : reg_alpha,                    # 디폴트 0 // L1 정규화 // 라쏘
        'n_estimators': int(round(n_estimators)),
        'gamma' : gamma,
        'colsample_bylevel' : colsample_bylevel
    }
    
    model = XGBRegressor(**params, early_stopping_rounds=20, eval_metric='rmse')
    model.fit(x_train, y_train,verbose=0,
               eval_set=[(x_test, y_test)],)
    y_pred = model.predict(x_test)
    result = r2_score(y_test, y_pred)

    return result


optimizer = BayesianOptimization(
    f = xgb_hamsu,
    pbounds=parameters,
    random_state=333
)
optimizer.maximize(init_points=20, n_iter=60)
best_param = optimizer.max
print('추천 조합 :', best_param)

# {'target': 0.8451236333707172, 
# 'params': {'colsample_bylevel': 1.0, 'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.1, 
# 'max_bin': 140.57744048728884, 'max_depth': 10.0, 'min_child_weight': 39.694383471319725, 
# 'n_estimators': 306.4416084802282, 'reg_alpha': 0.01, 'reg_lambda': 0.0001, 'subsample': 1.0}}

# {'target': 0.8484116129433522, 
# 'params': {'colsample_bylevel': 0.8625063898107251, 'colsample_bytree': 0.8745528592228077, 'gamma': 0.0, 'learning_rate': 0.1, 
# 'max_bin': 148.92297673940857, 'max_depth': 10.0, 'min_child_weight': 5.6469545855615655, 
# 'n_estimators': 366.5302720399443, 'reg_alpha': 0.0, 'reg_lambda': 55.63443717834303, 'subsample': 0.9521903492347671}}





