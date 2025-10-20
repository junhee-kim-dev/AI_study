from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#1. 데이터
dataset = load_iris()
# x = dataset.data
# y = dataset['target']
x, y = load_iris(return_X_y=True)

# print(x)
# print(y)
# print(x.shape, y.shape) #(150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42
)

#2. 모델구성
# from keras.models import Sequential
# from keras.layers import Dense

# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))


# 통상 ~~Regression 하면 회귀 모델이고, ~~Classifier 하면 분류 모델임

# model = LinearSVC(C=0.3)
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

#3.컴파일 훈련
# model.compile(loss='sparse_categorical_crossentropy', 
#               #sparse_categorical_crossentropy 원핫 안 해도 자동으로 해줌
#               optimizer='adam', 
#               metrics=['acc'], )

# model.fit(x_train, y_train, epochs=100, 
#           verbose=2, batch_size=32, )

# model.fit(x_train, y_train)

#4. 평가 예측
# loss = model.evaluate(x_test, y_test)
# loss = model.score(x_test, y_test)
# results = model.predict(x_test)

# print('ACC:', loss)     #LOSS: 0.2613928020000458
# print('ACC :', loss[1])     #ACC : 0.9736841917037964
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
models = [LinearSVC(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(f'{model} :', results)
    
# LinearSVC() : 1.0
# LogisticRegression() : 1.0
# DecisionTreeClassifier() : 1.0
# RandomForestClassifier() : 1.0
















































