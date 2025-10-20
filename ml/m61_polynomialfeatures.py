import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4, 2)
print(x)

pf = PolynomialFeatures(degree=2,)
x_pf = pf.fit_transform(pf)
print(x_pf)

#### 통상적으로
# 선형모델에 쓸 경우에는 include_bias=True 를 써서 1만 있는 컬럼을 만드는 게 좋음
# 왜냐하면 y = wx+b 의 bias=1 의 역할을 하기 떄문
# 비선형 모델 (rf, xgb 등)에 쓸 경우에는 include_bias = False가 좋음





