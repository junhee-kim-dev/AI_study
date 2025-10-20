# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
import datetime
import time

path = './Study25/_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

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
    
    model = XGBClassifier(**params, early_stopping_rounds=20, eval_metric='mlogloss')
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
optimizer.maximize(init_points=5, n_iter=10)
best_param = optimizer.max
print('추천 조합 :', best_param)