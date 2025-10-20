from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random

#1. 데이터
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

x = dataset.data    #(569, 30)
y = dataset.target  #(569,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, test_size=0.1, shuffle=True, 
    random_state=304, stratify=y)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

#2. 모델
model = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=100,
    random_state=333,
    # bootstrap=False,
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
results = model.score(x_test, y_test)
print('최종 점수:', results)

# 최종 점수: 0.9649122807017544

