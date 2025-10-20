import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning)

#1. 데이터
dataset = load_iris()
X = dataset.data
Y = dataset['target']

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, shuffle=True, random_state=42, train_size=0.8, stratify=Y
)

ss = StandardScaler()
ss.fit(x_train)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

n_split = 3
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)

#2. 모델구성
model = MLPClassifier()

#3. 훈련
score = cross_val_score(model, x_train, y_train, cv=kfold)
print('        acc :', score)
print('average acc :', round(np.mean(score), 5))
#         acc : [0.975 0.9   0.95 ]
# average acc : 0.94167

results = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_test)
print(results)
# [0 2 1 1 0 1 0 0 2 1 2 2 2 1 0 0 0 1 1 2 0 2 1 2 2 1 1 0 2 0]
# [0 2 1 1 0 1 0 0 2 1 2 2 2 1 0 0 0 1 1 2 0 2 1 2 2 2 1 0 2 0]

acc = accuracy_score(y_test, results)
print(acc)  # 0.9

















