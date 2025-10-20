from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np
import pandas as pd


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)
# print(y_train)
# print(x_train[0])
# print(y_train[0])   # 5

# print(x_train.shape, y_train.shape)     # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)       # (10000, 28, 28) (10000,)
# print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))

# print(pd.value_counts(y_train))
# 1    6742
# 7    6265
# 3    6131
# 2    5958
# 9    5949
# 0    5923
# 6    5918
# 8    5851
# 4    5842
# 5    5421

aaa = 645
print(y_train[aaa])

import matplotlib.pyplot as plt
plt.imshow(x_train[aaa], 'gray')
plt.show()
