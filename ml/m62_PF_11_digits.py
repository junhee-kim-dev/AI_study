from sklearn.datasets import load_digits

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import time, datetime
import matplotlib.pyplot as plt

# 시드 고정
seed = 123
import random
random.seed(seed)
np.random.seed(seed)
# tf.random.set_seed(seed)

datasets = load_digits()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
x_pf = pf.fit_transform(x)
print(x_pf)

x_ori = x.copy()

x_set = [x_ori, x_pf]
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

for id, i in enumerate(x_set):
        x_train, x_test, y_train, y_test = train_test_split(
            i, y, random_state=123, train_size=0.8,
        )

        model = XGBRegressor(random_state=123)

        model.fit(x_train, y_train)
        results = model.predict(x_test)

        r2 = r2_score(y_test, results)

        print(f"{id} 의 R2는 {r2:.4f}입니다!")