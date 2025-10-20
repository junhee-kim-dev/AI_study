import numpy as np

aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]]).T
# print(aaa.shape)
# (13, 2)

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=0.15)
outliers.fit(aaa)
results = outliers.fit(aaa)
print(results)

import matplotlib.pyplot as plt

def outlier2(data) :
    q1_list =[]
    q2_list =[]
    q3_list =[]
    iqr_list =[]
    lower_bound_list = []
    upper_bound_list = []
    for i in range(aaa.shape[1]) :
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

q1, q2, q3, iqr, lower, upper = outlier2(aaa)

print(q1)
print(q2)
print(q3)
print(iqr)
print(lower)
print(upper)

plt.figure(figsize=(14,10))
plt.title('before')
for i in range(aaa.shape[1]) :
    plt.subplot(1, aaa.shape[1], i+1)
    plt.boxplot(aaa[:,i])
    plt.axhline(upper[i], color='blue', label='upper bound')
    plt.axhline(lower[i], color='red', label='lower bound')
    plt.legend(loc='upper right')
    plt.grid()
plt.show()