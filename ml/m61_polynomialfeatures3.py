import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(12).reshape(4, 3)
print(x)

pf = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
x_pf = pf.fit_transform(x)
print(x_pf)

# [[ 0  1  2]
#  [ 3  4  5]
#  [ 6  7  8]
#  [ 9 10 11]]
# [[  1.   0.   1.   2.   0.   0.   0.   1.   2.   4.]
#  [  1.   3.   4.   5.   9.  12.  15.  16.  20.  25.]
#  [  1.   6.   7.   8.  36.  42.  48.  49.  56.  64.]
#  [  1.   9.  10.  11.  81.  90.  99. 100. 110. 121.]]
# [[  0.   1.   2.   0.   0.   2.]
#  [  3.   4.   5.  12.  15.  20.]
#  [  6.   7.   8.  42.  48.  56.]
#  [  9.  10.  11.  90.  99. 110.]]