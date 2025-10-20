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

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

seed = 72

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed
)

model4 = XGBRegressor(random_state=seed)
model4.fit(x_train, y_train)
print('3')
# print(f'# {model4.__class__.__name__}')
print(f'#    기존 R2 : {model4.score(x_test, y_test)}') 
# print('#', model4.feature_importances_)

# print('# 25%지점 :', np.percentile(model4.feature_importances_, 25))
per = np.percentile(model4.feature_importances_, 25)

col_names = []
# 삭제할 컬럼(25% 이하) 찾기
for i, fi in enumerate(model4.feature_importances_) :
    # print(i, fi)
    if fi <= per :
        col_names.append(dataset.feature_names[i])
    else :
        continue

x_f = pd.DataFrame(x, columns=dataset.feature_names)
x1 = x_f.drop(columns=col_names)
x2 = x_f[col_names]

x1_train, x1_test, x2_train, x2_test= train_test_split(
    x1, x2, train_size=0.8, random_state=seed
)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)

x_train = np.concatenate([x1_train, x2_train], axis=1)
x_test = np.concatenate([x1_test, x2_test], axis=1)

model4.fit(x_train, y_train)
score = model4.score(x_test, y_test)
print(f'#  새로운 R2 : {model4.score(x_test, y_test)}') 

#    기존 R2 : 0.4773898196707915
#  새로운 R2 : 0.4005918025622911