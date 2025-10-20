import numpy as np
import pandas as pd

data = pd.DataFrame(
    [[     2, np.nan,      6,      8,     10],
     [     2,      4, np.nan,      8, np.nan],
     [     2,      4,      6,      8,     10],
     [np.nan,      4, np.nan,      8, np.nan]]
).T
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = SimpleImputer(strategy='mean')
data2 = imputer.fit_transform(data)

imputer2 = SimpleImputer(strategy='median')
data3 = imputer2.fit_transform(data)

imputer3 = SimpleImputer(strategy='median')
data4 = imputer3.fit(data)

###########################################
data11 = pd.DataFrame(
    [[     2, np.nan,      6,      8,     10,  8],
     [     2,      4, np.nan,      8, np.nan,  4],
     [     2,      4,      6,      8,     10, 12],
     [np.nan,      4, np.nan,      8, np.nan,  8]]
).T
data11.columns = ['x1', 'x2', 'x3', 'x4']


imputer4 = SimpleImputer(strategy='most_frequent')
data5 = imputer4.fit_transform(data11)
print(data5)
# [[ 2.  2.  2.  8.]
#  [ 8.  4.  4.  4.]
#  [ 6.  4.  6.  8.]
#  [ 8.  8.  8.  8.]
#  [10.  4. 10.  8.]
#  [ 8.  4. 12.  8.]]

####################################################

imputer5 = SimpleImputer(strategy='constant', fill_value=777)
data6 = imputer5.fit_transform(data)
print(data6)
# [[  2.   2.   2. 777.]
#  [777.   4.   4.   4.]
#  [  6. 777.   6. 777.]
#  [  8.   8.   8.   8.]
#  [ 10. 777.  10. 777.]]

imputer6 = KNNImputer()
data7 = imputer6.fit_transform(data)
print(data7)
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]

########################################################

imputer7 = IterativeImputer()   # default = BayesianRide 회귀모델
data8 = imputer7.fit_transform(data)
print(data8)
# [[ 2.          2.          2.          2.0000005 ]
#  [ 4.00000099  4.          4.          4.        ]
#  [ 6.          5.99999928  6.          5.9999996 ]
#  [ 8.          8.          8.          8.        ]
#  [10.          9.99999872 10.          9.99999874]]









