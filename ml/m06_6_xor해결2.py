import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

#2. 모델구성
# model = Perceptron()              #구림
# model = LinearSVC()               #구림
# model = SVC()                     #된다.
model = DecisionTreeClassifier()    #된다.

#3. 컴파일, 훈련
model.fit(x_data, y_data)

#4. 평가
results = model.score(x_data, y_data)
# results = model.evaluate(x_data, y_data)
y_predict = model.predict(x_data)
acc = accuracy_score(y_data, np.round(y_predict))


print('model.score :', results)
print('accuracy    :', acc)

# model.score : 0.6931472420692444
# accuracy    : 0.5

# model.score : 6.891855446156114e-05
# accuracy    : 1.0

# model.score : 1.0
# accuracy    : 1.0

