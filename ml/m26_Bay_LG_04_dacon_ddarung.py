from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint

path ='./Study25/_data/dacon/따릉이/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'submission.csv')

train_csv = train_csv.fillna(train_csv.mean())
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']
print(x.shape)  #(1459, 9)
print(y.shape)  #(1459,)


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

# 추천 조합 : {'target': 0.8277613087210559, 
# 'params': {'colsample_bylevel': 0.7058119665390601, 'colsample_bytree': 0.8649168446478608, 
# 'learning_rate': 0.11232647627617029, 'max_bin': 137.12959843212332, 
# 'max_depth': 13.323072503115263, 'min_child_samples': 18.070530610185493, 
# 'min_child_weight': 7.683054800081146, 'min_split_gain': 4.025644961397025, 
# 'n_estimators': 468.34826554704404, 'num_leaves': 87.45187415762462, 
# 'reg_alpha': 1.0232292710037438, 'reg_lambda': 8.981534303488324, 
# 'subsample': 0.7360193524791477}}