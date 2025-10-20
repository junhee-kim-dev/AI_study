import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import random
import tensorflow as tf

#<random 값 고정>   gpu 학습을 하면 random을 고정하더라도 값이 틀어질 수 있음.
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

#1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

#2. 모델구성
# model = Perceptron()
# model = LinearSVC()
model = Sequential()
model.add(Dense(10000, input_dim=2, activation='sigmoid'))
# model.add(Dense(5000))
# model.add(Dense(400))
# model.add(Dense(300))
# model.add(Dense(300))
# model.add(Dense(300))
# model.add(Dense(300))
# model.add(Dense(300))
# model.add(Dense(300))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, epochs=2000)

#4. 평가
# results = model.score(x_data, y_data)
results = model.evaluate(x_data, y_data)
y_predict = model.predict(x_data)
acc = accuracy_score(y_data, np.round(y_predict))


print('model.score :', results[0])
print('accuracy    :', acc)

# model.score : 0.6931472420692444
# accuracy    : 0.5

# model.score : 6.891855446156114e-05
# accuracy    : 1.0



