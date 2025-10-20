
from keras.datasets import cifar100
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

(x_trn, y_trn), (x_tst,y_tst) = cifar100.load_data()

x_trn = (x_trn-127.5)/(510.0)
x_tst = (x_tst-127.5)/(510.0)


x_trn = x_trn.reshape(-1,x_trn.shape[1]*x_trn.shape[2]*x_trn.shape[3])
x_tst = x_tst.reshape(-1,x_tst.shape[1]*x_tst.shape[2]*x_tst.shape[3])
# print(x_trn.shape, x_tst.shape)     # (50000, 3072) (10000, 3072)

# exit()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


lda = LinearDiscriminantAnalysis(n_components=99)
x_train_1 = lda.fit_transform(x_trn, y_trn)
x_test_1 = lda.transform(x_tst)

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train_1, y_trn)
score = xgb.score(x_test_1, y_tst)

print(f'n_components_99 :',score)    

print(np.cumsum(lda.explained_variance_ratio_))


# n_components_99 : 0.1211

# [0.10131909 0.16234557 0.20716146 0.24077165 0.26807672 0.29247064
#  0.31560461 0.33730533 0.35730029 0.37557165 0.39251105 0.40847786
#  0.4236487  0.43852258 0.45236242 0.46575303 0.47876778 0.49157994
#  0.50383965 0.51566977 0.52743736 0.53865189 0.54980662 0.56038868
#  0.57070134 0.58074026 0.59045537 0.59993683 0.60923803 0.61851878
#  0.62758507 0.63660162 0.64559419 0.65434984 0.66274234 0.67107041
#  0.67928057 0.68739823 0.69532181 0.70309141 0.71063001 0.71793137
#  0.72518129 0.73233199 0.73944925 0.74648669 0.75340438 0.76023243
#  0.76684831 0.77337452 0.77977866 0.7860676  0.79224825 0.79833689
#  0.80436018 0.81033232 0.81610919 0.82187513 0.82755161 0.83315036
#  0.83867396 0.84412624 0.84943898 0.85468907 0.85991716 0.86510843
#  0.87023156 0.87526325 0.88027371 0.88519221 0.89007686 0.89495224
#  0.89977871 0.90456023 0.90927684 0.91387831 0.91846196 0.92298982
#  0.9274065  0.93179817 0.93610791 0.94037533 0.94452347 0.94864824
#  0.95271325 0.95667579 0.96057646 0.96439043 0.96808131 0.97174884
#  0.97537265 0.97889967 0.98238902 0.98575492 0.98901577 0.99208888
#  0.99506245 0.99772778 1.        ]