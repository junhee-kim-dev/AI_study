import numpy as np

aaa = np.array([-10,  2,  3,  4,  5,  6,  7,
                -11,  -2,  -3,  -4, - 5,  -6,  -7,
                  18,  19, 20, 21, 22, 35,
                  8,  9, 10, 11, 12, 50])
bbb = np.array([-13,  42,  23,  34, 25,  36,  27,
                -11,  -2,  -13,  -24, - 5,  -16,  -17,
                187,  29, 10, 1, 2, 35,
                8,  9, 10, 11, 12, 50])

def outlier(data) :
    quantile_1, quantile_2, quantile_3 = np.percentile(data, [25,50,75])
    print(f"1사분위: {quantile_1}")
    print(f"2사분위: {quantile_2}")
    print(f"3사분위: {quantile_3}")
    iqr = quantile_3 - quantile_1
    print(f"   IQR : {iqr}")
    lower_bound = quantile_1 - (iqr*1.5)
    upper_bound = quantile_3 + (iqr*1.5)
    
    return np.where((data > upper_bound) | (data < lower_bound)) , iqr, lower_bound, upper_bound

outlier_loc, iqr, lower, upper = outlier(aaa)
outlier_locb, iqrb, lowerb, upperb = outlier(bbb)
print('이상치의 위치:', outlier_loc)

# 1사분위: 4.0
# 2사분위: 7.0
# 3사분위: 10.0
#    IQR : 6.0
# 이상치의 위치: (array([ 0, 12]),)

import matplotlib.pyplot as plt
plt.boxplot([aaa, bbb])
# plt.boxplot(bbb)
plt.axhline(upper, color='red', label='upper bound')
plt.axhline(upperb, color='blue', label='upper bound')
plt.axhline(lower, color='red', label='lower bound')
plt.axhline(lowerb, color='blue', label='lower bound')
plt.legend()
plt.grid()
plt.show()


















