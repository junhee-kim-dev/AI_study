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
xgb = XGBClassifier()
# from sklearn.model_selection import RandomizedSearchCV
# model = RandomizedSearchCV(xgb, parameters, cv=kfold,
#                      verbose=1,
#                      refit=True,
#                      n_jobs=18,
#                      random_state=333,
#                      )

# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import HalvingGridSearchCV

# model = HalvingGridSearchCV(xgb, parameters, cv=kfold,
#                         verbose=1,
#                         refit=True,
#                         n_jobs=18,
#                         factor=factor,               # 배율 (min_resources * 3) *3 ...
#                         min_resources=min_resources,       # 최소 훈련량
#                         )

import math
print(x_train.shape)
factor = 3
n_iterations = 3
min_resources = max(1, math.floor(x_train.shape[0] // (factor ** (n_iterations - 1))))

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

model = HalvingRandomSearchCV(xgb, parameters, cv=kfold,
                          verbose=1,
                          refit=True,
                          n_jobs=18,
                          factor=factor, 
                          min_resources=min_resources,
                          )

#3. 훈련
s_time = time.time()
model.fit(x_train, y_train)
e_time = time.time()

print('      best_variable :', model.best_estimator_)
print('        best_params :', model.best_params_)

#4. 평가
print('         best_score :', model.best_score_)
print('   model_best_score :', model.score(x_test, y_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

y_pred_best = model.best_estimator_.predict(x_test)
acc_best = accuracy_score(y_test, y_pred_best)
print('best_accuracy_score :', acc_best)
print('     accuracy_score :', acc)
print('       running_time :', np.round(e_time - s_time, 3), 'sec')

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

path = './Study25/_save/ml/m22/'

joblib.dump(model.best_estimator_, path + 'm22_HalvingRandomSearch_00.joblib')

#         best_params : {'max_depth': 6, 'learning_rate': 0.01}
#          best_score : 0.9304347826086957
#    model_best_score : 0.9
# best_accuracy_score : 0.9
#      accuracy_score : 0.9
#        running_time : 3.082 sec

#         best_params : {'min_child_weight': 2, 'learning_rate': 0.01}
#          best_score : 0.9304347826086957
#    model_best_score : 0.8666666666666667
# best_accuracy_score : 0.8666666666666667
#      accuracy_score : 0.8666666666666667
#        running_time : 2.719 sec