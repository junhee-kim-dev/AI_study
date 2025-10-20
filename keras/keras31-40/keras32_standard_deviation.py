from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import datetime

data = np.array([[ 1, 2, 3,  1],
                 [ 4, 5, 6,  2],
                 [ 7, 8, 9,  3],
                 [10,11,12,114],
                 [13,14,15,115]])
print(data.shape)                   # (5, 4)

# 1) 평균
means = np.mean(data, axis=0)
print('         평균:', means)                # [ 7.  8.  9. 47.]

# 2) 모집단 분산(n으로 나눔)
population_variances = np.var(data, axis=0)
print('  모집단 분산:', population_variances) # [  18.   18.   18. 3038.]

# 3) 표본 분산(n-1으로 나눔)
variance = np.var(data, axis=0, ddof=1)       # ddof = n-1로 나누겠다는 뜻
print('    표본 분산:', variance)             # [  22.5   22.5   22.5 3797.5]

# 4) 표본 표준편차
std1 = np.std(data, axis=0, ddof=1)
print('표본 표준편차:', std1)                 # [ 4.74341649  4.74341649  4.74341649 61.62385902]

# 5) 모집단 표준편자
std2 = np.std(data, axis=0)
print('    표준 편차:', std2)                 # [ 4.24264069  4.24264069  4.24264069 55.11805512]

# 6) StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print('StandardScaler : \n', scaled_data)
# StandardScaler :
#  [[-1.41421356 -1.41421356 -1.41421356 -0.83457226]
#  [-0.70710678 -0.70710678 -0.70710678 -0.81642939]
#  [ 0.          0.          0.         -0.79828651]
#  [ 0.70710678  0.70710678  0.70710678  1.21557264]
#  [ 1.41421356  1.41421356  1.41421356  1.23371552]]