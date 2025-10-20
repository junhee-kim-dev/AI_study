
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 시드 고정
seed = 123
import random
random.seed(seed)
np.random.seed(seed)

path = './Study25/_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

test_csv = test_csv.replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

x = train_csv.drop(['Outcome'], axis=1)
zero_na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
x[zero_na_columns] = x[zero_na_columns].replace(0, np.nan)
x = x.fillna(x.mean())
y = train_csv['Outcome']

x = x.values.copy()
y = y.values.copy()

import matplotlib.pyplot as plt

def outlier2(data) :
    q1_list =[]
    q2_list =[]
    q3_list =[]
    iqr_list =[]
    lower_bound_list = []
    upper_bound_list = []
    for i in range(x.shape[1]) :
        quantile_1, quantile_2, quantile_3 = np.percentile(data[:,i], [25,50,75])
        q1_list.append(quantile_1)
        q2_list.append(quantile_2)
        q3_list.append(quantile_3)
        print(f"1사분위: {quantile_1}")
        print(f"2사분위: {quantile_2}")
        print(f"3사분위: {quantile_3}")
        iqr = quantile_3 - quantile_1
        iqr_list.append(iqr)
        print(f"   IQR : {iqr}")
        lower_bound = quantile_1 - (iqr*1.5)
        upper_bound = quantile_3 + (iqr*1.5)
        
        lower_bound_list.append(lower_bound)
        upper_bound_list.append(upper_bound)

    return np.array(q1_list), np.array(q2_list), np.array(q3_list), \
        np.array(iqr_list), np.array(lower_bound_list), np.array(upper_bound_list)

q1, q2, q3, iqr, lower, upper = outlier2(x)

print(q1)
print(q2)
print(q3)
print(iqr)
print(lower)
print(upper)

plt.figure(figsize=(14,10))
plt.title('before')
for i in range(x.shape[1]) :
    plt.subplot(1, x.shape[1], i+1)
    plt.boxplot(x[:,i])
    plt.axhline(upper[i], color='blue', label='upper bound')
    plt.axhline(lower[i], color='red', label='lower bound')
    plt.legend(loc='upper right')
    plt.grid()
    

for i in range(x.shape[1]) :
    for j in range(len(x)) :
        if x[j,i] > upper[i] :
            x[j,i] = upper[i]
        elif  x[j,i] < lower[i] :
            x[j,i] = lower[i]
        else :
            continue

plt.figure(figsize=(14,10))
plt.title('after')
for i in range(x.shape[1]) :
    plt.subplot(1, x.shape[1], i+1)
    plt.boxplot(x[:,i])
    plt.axhline(upper[i], color='blue', label='upper bound')
    plt.axhline(lower[i], color='red', label='lower bound')
    plt.legend(loc='upper right')
    plt.grid()
    
plt.show()

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score, accuracy_score, f1_score
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8,
)

model = XGBClassifier(random_state=123)

model.fit(x_train, y_train)
results = model.predict(x_test)

r2 = accuracy_score(y_test, results)
f1 = f1_score(y_test, results)

print(f"ACC는 {r2:.4f}입니다!")
print(f"F1는 {f1:.4f}입니다!")

# ACC는 0.7023입니다!
# F1는 0.5895입니다!