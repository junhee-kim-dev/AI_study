import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV    # 채로 걸러서 파라미터를 찾아내고 cross_val까지 하겠다.

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8, stratify=y
)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

parameters = [
    {'C':[1,10,100,1000],
     'kernel':['linear', 'sigmoid'],
     'degree':[3,4,5]
     }, #24번
    {'C':[1,10,100],
     'kernel':['rbf'],
     'gamma':[0.001,0.0001]
     }, #6번
    {'C':[1,10,100,1000],
     'kernel':['sigmoid'],
     'gamma':[0.01,0.001,0.0001],
     'degree':[3,4]
     }  #24번
]  

#2. 모델
model = GridSearchCV(SVC(), parameters, cv=kfold,
                     verbose=1,
                     refit=True,    # 1회
                     n_jobs=18,
                     )   # (24 + 6 + 24) * n_split + refit(1) = 271회

#3. 훈련
import time
s_time = time.time()
model.fit(x_train, y_train)
e_time = time.time()

print('    best_variable :', model.best_estimator_)
print('      best_params :', model.best_params_)

#4. 평가
print('       best_score :', model.best_score_)
print(' model_best_score :', model.score(x_test, y_test))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('   accuracy_score :', acc)
print('     running_time :', np.round(e_time - s_time, 3), 'sec')

#     best_variable : SVC(C=10, kernel='linear')
#       best_params : {'C': 10, 'degree': 3, 'kernel': 'linear'}
#        best_score : 0.9916666666666668
#  model_best_score : 0.9
#    accuracy_score : 0.9
#      running_time : 0.76 sec

