import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

#1. 데이터
x,y = load_iris(return_X_y=True)

n_split=5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)

#2. 모델
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)
model = MLPClassifier()
scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))


# acc : [0.96666667 0.96666667 1.         0.96666667 0.93333333] 
# 평균 acc : 0.9667