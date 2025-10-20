
from keras.datasets import cifar10
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPool2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model, load_model
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time

(x_trn, y_trn), (x_tst,y_tst) = cifar10.load_data()

x_trn = (x_trn-127.5)/(510.0)
x_tst = (x_tst-127.5)/(510.0)

x_trn = x_trn.reshape(-1,x_trn.shape[1]*x_trn.shape[2]*x_trn.shape[3])
x_tst = x_tst.reshape(-1,x_tst.shape[1]*x_tst.shape[2]*x_tst.shape[3])
# print(x_trn.shape, x_tst.shape)     # (50000, 3072) (10000, 3072)

# exit()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

for i in range(9) :
    lda = LinearDiscriminantAnalysis(n_components=i+1)
    x_train_1 = lda.fit_transform(x_trn, y_trn)
    x_test_1 = lda.transform(x_tst)

    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.fit(x_train_1, y_trn)
    score = xgb.score(x_test_1, y_tst)

    print(f'n_components_{i+1} :',score)    

print(np.cumsum(lda.explained_variance_ratio_))

# n_components_1 : 0.1845
# n_components_2 : 0.2601
# n_components_3 : 0.3085
# n_components_4 : 0.3298
# n_components_5 : 0.336
# n_components_6 : 0.3502
# n_components_7 : 0.3596
# n_components_8 : 0.3664
# n_components_9 : 0.3679





