import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import warnings

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
model = GridSearchCV(xgb, parameters, cv=kfold,
                     verbose=1,
                     refit=True,
                     n_jobs=18,
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

# y_pred_best = model.best_estimator_.predict(x_test)
# acc_best = accuracy_score(y_test, y_pred_best)
# print('best_accuracy_score :', acc_best)
print('     accuracy_score :', acc)
print('       running_time :', np.round(e_time - s_time, 3), 'sec')

#              best_params : {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 500}
#               best_score : 0.95
#         model_best_score : 0.9
#           accuracy_score : 0.9
#             running_time : 3.079 sec

#               best_score : 0.95
#         model_best_score : 0.9
#      best_accuracy_score : 0.9     # 아래 그냥 acc 랑 똑같음 그래서 best_estimator 사장됨
#           accuracy_score : 0.9
#             running_time : 3.21 sec

# Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
#        'param_learning_rate', 'param_max_depth', 'param_n_estimators',
#        'param_min_child_weight', 'params', 'split0_test_score',
#        'split1_test_score', 'split2_test_score', 'split3_test_score',
#        'split4_test_score', 'mean_test_score', 'std_test_score',
#        'rank_test_score'],
#       dtype='object')

import pandas as pd

best_csv = pd.DataFrame(model.cv_results_).sort_values(['rank_test_score'], ascending=True)
print(best_csv.columns)

path = './Study25/_save/ml/'
best_csv.to_csv(path + 'm15/cv_results_00.csv', index=False)



















































