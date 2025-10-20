# https://www.kaggle.com/competitions/playground-series-s4e1/submissions

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import f1_score
from keras.layers import BatchNormalization
from keras.layers import Dropout
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = './Study25/_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

train_csv[['Tenure', 'Balance']] = train_csv[['Tenure', 'Balance']].replace(0, np.nan)
train_csv = train_csv.fillna(train_csv.mean())

test_csv[['Tenure', 'Balance']] = test_csv[['Tenure', 'Balance']].replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

oe = OrdinalEncoder()       # 이렇게 정의 하는 것을 인스턴스화 한다고 함
oe.fit(train_csv[['Geography', 'Gender']])
train_csv[['Geography', 'Gender']] = oe.transform(train_csv[['Geography', 'Gender']])
test_csv[['Geography', 'Gender']] = oe.transform(test_csv[['Geography', 'Gender']])

train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']



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
#        best_score : 0.8655653630790878
#  model_best_score : 0.8652710031205502
#    accuracy_score : 0.8652710031205502
#      running_time : 18.733 sec

#       best_params : {'learning_rate': 0.1, 'min_child_weight': 5}
#        best_score : 0.8655021568173128
#  model_best_score : 0.865725452176811
#    accuracy_score : 0.865725452176811
#      running_time : 22.576 sec

# 추천 조합 : {'target': 0.8694861851672322, 'params': {'colsample_bylevel': 0.5, 'colsample_bytree': 1.0, 
# 'gamma': 0.0, 'learning_rate': 0.1, 'max_bin': 500.0, 'max_depth': 5.46000829650013, 'min_child_weight': 1.0, 
# 'n_estimators': 294.3943627279733, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 1.0}}