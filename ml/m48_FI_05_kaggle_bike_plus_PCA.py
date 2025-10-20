from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
from keras.callbacks import EarlyStopping

path = './Study25/_data/kaggle/bike-sharing-demand/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['count'], axis=1)
y = train_csv[['count']]


import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

seed = 72

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed
)

model4 = XGBRegressor(random_state=seed)
model4.fit(x_train, y_train)
print('5')
# print(f'# {model4.__class__.__name__}')
print(f'# R2 : {model4.score(x_test, y_test)}') 
# print('#', model4.feature_importances_)

# print('# 25%지점 :', np.percentile(model4.feature_importances_, 25))
per = np.percentile(model4.feature_importances_, 25)

col_names = []
# 삭제할 컬럼(25% 이하) 찾기
for i, fi in enumerate(model4.feature_importances_) :
    # print(i, fi)
    if fi <= per :
        col_names.append(x.columns[i])
    else :
        continue


x_f = pd.DataFrame(x, columns=x.columns)
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
print(f'# new R2 : {model4.score(x_test, y_test)}') 

# R2 : 0.9993132948875427
# new R2 : 0.9992846846580505