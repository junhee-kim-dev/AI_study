# dacon, 데이터 파일 별도
# https://dacon.io/competitions/official/236068/leaderboard

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = './Study25/_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_na_columns] = x[zero_na_columns].replace(0, np.nan)
x = x.fillna(x.mean())
y = train_csv['Outcome']


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from xgboost import XGBRegressor, XGBClassifier
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
    
    model = XGBClassifier(**params, early_stopping_rounds=20, eval_metric='rmse')
    model.fit(x_train, y_train,verbose=0,
               eval_set=[(x_test, y_test)],)
    y_pred = model.predict(x_test)
    result = accuracy_score(y_test, y_pred)

    return result


optimizer = BayesianOptimization(
    f = xgb_hamsu,
    pbounds=parameters,
    random_state=333
)
optimizer.maximize(init_points=20, n_iter=60)
best_param = optimizer.max
print('추천 조합 :', best_param)

#       best_params : {'max_depth': 6, 'learning_rate': 0.1}
#        best_score : 0.7428571428571429
#  model_best_score : 0.7938931297709924
#    accuracy_score : 0.7938931297709924
#      running_time : 0.362 sec

#       best_params : {'learning_rate': 0.01, 'max_depth': 12, 'n_estimators': 500}
#        best_score : 0.7436322101656196
#  model_best_score : 0.8015267175572519
#    accuracy_score : 0.8015267175572519
#      running_time : 3.37 sec

# 추천 조합 : {'target': 0.8484848484848485, 'params': {'colsample_bylevel': 0.9351363749485913, 
# 'colsample_bytree': 0.5153219497073163, 'gamma': 3.5110371841368067, 'learning_rate': 0.01766972512018996,
# 'max_bin': 407.4329337781983, 'max_depth': 6.671235945089808, 'min_child_weight': 5.194886053413886, 
# 'n_estimators': 375.02678109659786, 'reg_alpha': 0.7900779146758996, 'reg_lambda': 73.7128662320576, 
# 'subsample': 0.6581459916666714}}

