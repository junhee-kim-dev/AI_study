import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings
import joblib

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8, stratify=y
)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import time

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

parameters = [
    {'n_estimators' : [100, 500], 'max_depth' : [6,10,12], 'learning_rate' : [0.1, 0.01, 0.001]},
    {'max_depth' : [6,8,10,12], 'learning_rate' : [0.1, 0.01, 0.001]},
    {'min_child_weight' : [2,3,5,10], 'learning_rate' : [0.1, 0.01, 0.001]}
]  

#2. 모델
# xgb = XGBClassifier()
# model = GridSearchCV(xgb, parameters, cv=kfold,
#                      verbose=1,
#                      refit=True,
#                      n_jobs=18,
#                      )

import joblib
path = './Study25/_save/ml/m22/'
model = joblib.load(path + 'm22_HalvingRandomSearch_00.joblib')
print(model)
print(type(model))
'''XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              feature_weights=None, gamma=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.1, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=6,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, multi_strategy=None, n_estimators=500,
              n_jobs=None, num_parallel_tree=None, ...)'''
# <class 'xgboost.sklearn.XGBClassifier'>

#3. 훈련
# s_time = time.time()
# model.fit(x_train, y_train)
# e_time = time.time()

# print('      best_variable :', model.best_estimator_)
# print('        best_params :', model.best_params_)

# #4. 평가
# print('         best_score :', model.best_score_)
# print('   model_best_score :', model.score(x_test, y_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

# y_pred_best = model.best_estimator_.predict(x_test)
# acc_best = accuracy_score(y_test, y_pred_best)
# print('  best_accuracy_score :', acc_best)
print('       accuracy_score :', acc)          # accuracy_score : 0.9

# print('         running_time :', np.round(e_time - s_time, 3), 'sec')

#         best_params : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}
#          best_score : 0.95
#    model_best_score : 0.9
#      accuracy_score : 0.9
#        running_time : 3.079 sec

#          best_score : 0.95
#    model_best_score : 0.9
# best_accuracy_score : 0.9     # 아래 그냥 acc 랑 똑같음 그래서 best_estimator 사장됨
#      accuracy_score : 0.9
#        running_time : 3.21 sec


# joblib.dump(model.best_estimator_, path + 'm16_best_model.joblib')

