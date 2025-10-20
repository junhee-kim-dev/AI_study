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

# 추천 조합 : {'target': 0.5484799015138417, 'params': {'colsample_bylevel': 0.5248095876093077, 
# 'colsample_bytree': 0.5368790106039989, 'gamma': 1.905761938550206, 'learning_rate': 0.049969307899729046, 
# 'max_bin': 88.07532351401935, 'max_depth': 7.577399419899368, 'min_child_weight': 8.63077756305184, 
# 'n_estimators': 483.2339153406278, 'reg_alpha': 4.314713851012508, 'reg_lambda': 16.36607417087248, 
# 'subsample': 0.7857578420117002}}