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

datasets = load_digits()
x = datasets.data
y = datasets.target

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

seed = 72

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model4 = XGBClassifier(random_state=seed)
model4.fit(x_train, y_train)
print('11')
print(f'# {model4.__class__.__name__}')
print(f'# ACC : {model4.score(x_test, y_test)}') 
print('#', model4.feature_importances_)

print('# 25%지점 :', np.percentile(model4.feature_importances_, 25))
per = np.percentile(model4.feature_importances_, 25)

col_names = []
# 삭제할 컬럼(25% 이하) 찾기
for i, fi in enumerate(model4.feature_importances_) :
    # print(i, fi)
    if fi <= per :
        col_names.append(datasets.feature_names[i])
    else :
        continue

x = pd.DataFrame(x, columns=datasets.feature_names)
x = x.drop(columns=col_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model4.fit(x_train, y_train)
print(f'# ACC2 : {model4.score(x_test, y_test)}')

# XGBClassifier
# ACC : 0.9638888888888889
# [0.         0.0417367  0.00713317 0.00617588 0.00471327 0.03324919
#  0.01295262 0.01797988 0.         0.0098784  0.0138523  0.00355001
#  0.00664873 0.01145378 0.00484313 0.02916133 0.         0.00360038
#  0.00421722 0.03503135 0.01187184 0.04732806 0.00412742 0.
#  0.         0.00634597 0.02831241 0.00830524 0.03511441 0.01669818
#  0.00995733 0.         0.         0.05678235 0.00633535 0.00731945
#  0.06252539 0.02139928 0.02870381 0.         0.         0.01168852
#  0.03576131 0.03981652 0.01547159 0.01254679 0.03446812 0.
#  0.         0.01564377 0.00569895 0.00788061 0.01069538 0.01020297
#  0.02618384 0.00173473 0.         0.00335314 0.01054431 0.00435389
#  0.07735547 0.01514944 0.03808966 0.02605717]
# 25%지점 : 0.003995659702923149
# ACC2 : 0.9694444444444444