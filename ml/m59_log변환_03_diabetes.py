from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from keras.callbacks import EarlyStopping

dataset = load_diabetes()
x = dataset.data
y = dataset.target



x_ori = x.copy()
x_log = np.log1p(x)

y_ori = y.copy()
y_log = np.log1p(y)
print(np.max(y))
# 1.7917611358933327

x_set = [x_ori, x_log]
y_set = [y_ori, y_log]
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

for id, i in enumerate(x_set):
    for id_2, j in enumerate(y_set):
        x_train, x_test, y_train, y_test = train_test_split(
            i, j, random_state=123, train_size=0.8,
        )

        model = XGBRegressor(random_state=123)

        model.fit(x_train, y_train)
        results = model.predict(x_test)

        r2 = r2_score(y_test, results)

        print(f"{id}/{id_2}의 R2는 {r2:.4f}입니다!")
        
# 0/0의 R2는 0.3907입니다!
# 0/1의 R2는 0.4783입니다!
# 1/0의 R2는 0.3907입니다!
# 1/1의 R2는 0.4783입니다!