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
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8,
)

model = XGBRegressor(random_state=123)

model.fit(x_train, y_train)
results = model.predict(x_test)

r2 = r2_score(y_test, results)

print(f"R2는 {r2:.4f}입니다!")

# 0 의 R2는 0.3907입니다!
# 1 의 R2는 0.3807입니다!
# R2는 0.4454입니다!