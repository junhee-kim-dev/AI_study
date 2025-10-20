from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
from keras.callbacks import EarlyStopping

path = './Study25/_data/kaggle/bike-sharing-demand/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['count'], axis=1)
y = train_csv[['count']]


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

# 추천 조합 : {'target': 0.9996800893361789, 
# 'params': {'colsample_bylevel': 0.5357585280731932, 'colsample_bytree': 0.9977838092970557, 
# 'learning_rate': 0.06439121736138213, 'max_bin': 459.2205930125164, 
# 'max_depth': 5.94272168007946, 'min_child_samples': 15.966233526942638, 
# 'min_child_weight': 5.922427371327443, 'min_split_gain': 3.5413096922645124, 
# 'n_estimators': 347.5497426870524, 'num_leaves': 35.40592195882105, 
# 'reg_alpha': 0.03720511157856099, 'reg_lambda': 5.694247198573278, 
# 'subsample': 0.7409201554412139}}


