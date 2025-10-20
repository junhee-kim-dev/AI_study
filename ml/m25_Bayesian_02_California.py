import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler

#1. 데이터
dataset = fetch_california_housing()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, shuffle=True, random_state=304)

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
import time

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

parameters ={
     'learning_rate' : (0.001, 0.1),
     'max_depth' : (3, 10),
     'min_child_weight' : (1,50),
     'subsample' : (0.5,1),
     'colsample_bytree' : (0.5,1),
     'max_bin' : (9,500),
     'reg_lambda' : (1e-4,10),
     'reg_alpha' : (0.01,50)
     }  

#2. 모델
def bayesian(learning_rate, max_depth, min_child_weight, 
             subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha) :
    model = XGBRegressor(learning_rate = learning_rate,
                         max_depth = int(max_depth),
                         min_child_weight = int(min_child_weight),
                         subsample = subsample,
                         colsample_bytree = colsample_bytree,
                         max_bin = int(max_bin),
                         reg_lambda = reg_lambda,
                         reg_alpha = reg_alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    return 1-r2


#       best_params : {'min_child_weight': 2, 'learning_rate': 0.1}
#        best_score : 0.8296796187167621
#  model_best_score : 0.8275176783608397
#    accuracy_score : 0.8275176783608397
#      running_time : 26.946 sec

#       best_params : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}
#        best_score : 0.8461942627016722
#  model_best_score : 0.8440473231598093
#    accuracy_score : 0.8440473231598093
#      running_time : 27.202 sec
from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = bayesian,
    pbounds=parameters,
    random_state=333
)

optimizer.maximize(init_points=5, n_iter=20)
best_param = optimizer.max

raw_param = best_param['params']
int_keys = ['max_depth', 'min_child_weight', 'max_bin']
def intgerization(k, v):
    return int(round(v)) if k in int_keys else v

clean_params = {k: intgerization(k, v) for k, v in raw_param.items()}

print(clean_params)
