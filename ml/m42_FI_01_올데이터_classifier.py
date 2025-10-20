from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np

seed = 123
random.seed(seed)
np.random.seed(seed)

data1 = load_breast_cancer()
data2 = load_wine()
data3 = load_digits()

datasets = [data1, data2, data3]
dataname = ['Cancer', 'Wine', 'Digits']

model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)

models = [model1, model2, model3, model4]

for i, data in enumerate(datasets):
    x = data.data
    y = data.target
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=seed, stratify=y
    )
    
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    
    print(f'############ {dataname[i]} ############')

    for model in models :
        model.fit(x_train, y_train)
        print("==========", model.__class__.__name__, "==========")
        print('acc :', model.score(x_test, y_test))
        print(np.round(model.feature_importances_,4))


# ############ Cancer ############
# ========== DecisionTreeClassifier ==========
# acc : 0.9230769230769231
# [0.000e+00 6.120e-02 1.150e-02 1.900e-03 0.000e+00 1.420e-02 1.200e-02
#  0.000e+00 0.000e+00 6.700e-03 2.300e-03 0.000e+00 0.000e+00 0.000e+00
#  0.000e+00 8.000e-03 0.000e+00 5.000e-03 0.000e+00 0.000e+00 0.000e+00
#  5.000e-04 6.922e-01 4.460e-02 4.700e-03 0.000e+00 1.740e-02 1.176e-01
#  0.000e+00 0.000e+00]
# ========== RandomForestClassifier ==========
# acc : 0.958041958041958
# [0.0373 0.0162 0.0531 0.0317 0.0043 0.0155 0.0501 0.1209 0.0033 0.0027
#  0.0093 0.0069 0.0136 0.045  0.0062 0.0045 0.0036 0.0038 0.003  0.0071
#  0.1053 0.0155 0.1454 0.1008 0.0126 0.0137 0.0274 0.1192 0.0154 0.0066]
# ========== GradientBoostingClassifier ==========
# acc : 0.9790209790209791
# [2.600e-03 1.900e-02 2.400e-03 1.000e-04 2.300e-03 1.500e-03 1.100e-03
#  3.750e-02 1.800e-03 8.000e-04 4.100e-03 8.900e-03 2.000e-04 4.100e-03
#  1.900e-03 2.300e-03 2.400e-03 7.000e-04 4.000e-04 1.800e-03 1.080e-02
#  3.570e-02 4.087e-01 3.067e-01 1.190e-02 1.300e-03 1.660e-02 1.087e-01
#  1.400e-03 2.300e-03]
# ========== XGBClassifier ==========
# acc : 0.9790209790209791
# [0.     0.0266 0.0229 0.0206 0.0016 0.0021 0.0154 0.0394 0.0019 0.007
#  0.0078 0.     0.0053 0.0095 0.007  0.0047 0.0095 0.0235 0.0029 0.0047
#  0.0207 0.0144 0.5421 0.0247 0.0176 0.0137 0.0206 0.1221 0.0041 0.0076]
# ############ Wine ############
# ========== DecisionTreeClassifier ==========
# acc : 0.9111111111111111
# [0.     0.     0.0223 0.     0.     0.     0.408  0.     0.     0.3891
#  0.0222 0.0214 0.137 ]
# ========== RandomForestClassifier ==========
# acc : 0.9777777777777777
# [0.1421 0.0266 0.0205 0.0226 0.0339 0.0644 0.1495 0.008  0.0341 0.142
#  0.065  0.1405 0.1507]
# ========== GradientBoostingClassifier ==========
# acc : 0.9333333333333333
# [1.250e-02 3.550e-02 8.600e-03 8.700e-03 0.000e+00 0.000e+00 3.086e-01
#  2.000e-04 3.700e-03 2.662e-01 1.180e-02 6.650e-02 2.778e-01]
# ========== XGBClassifier ==========
# acc : 0.9777777777777777
# [0.0798 0.089  0.0094 0.0038 0.025  0.0295 0.1973 0.0084 0.026  0.2627
#  0.0347 0.0783 0.1561]
# ############ Digits ############
# ========== DecisionTreeClassifier ==========
# acc : 0.8644444444444445
# [0.     0.     0.0086 0.0105 0.0031 0.0687 0.0016 0.     0.0016 0.0138
#  0.0048 0.0051 0.0569 0.0185 0.0008 0.     0.     0.0088 0.0092 0.018
#  0.0038 0.0877 0.0024 0.     0.0016 0.0045 0.0492 0.0508 0.0081 0.0228
#  0.013  0.     0.     0.0513 0.014  0.0033 0.0758 0.0278 0.     0.
#  0.     0.0014 0.1203 0.0537 0.0019 0.0065 0.0073 0.     0.     0.
#  0.0123 0.0049 0.005  0.0122 0.0236 0.     0.     0.0023 0.0171 0.0113
#  0.0609 0.0049 0.0025 0.0057]
# ========== RandomForestClassifier ==========
# acc : 0.9822222222222222
# [0.     0.002  0.0243 0.0102 0.0109 0.0187 0.0085 0.0013 0.0001 0.0117
#  0.0248 0.0067 0.0198 0.0259 0.0069 0.0008 0.     0.0067 0.0199 0.0231
#  0.0305 0.0525 0.0109 0.0002 0.0001 0.0146 0.0396 0.0242 0.0267 0.0207
#  0.0259 0.     0.     0.0294 0.0303 0.0183 0.0419 0.021  0.0257 0.
#  0.0001 0.0111 0.0381 0.0425 0.0221 0.0217 0.0172 0.     0.0001 0.0023
#  0.018  0.021  0.0134 0.0238 0.0252 0.0024 0.     0.0022 0.0222 0.0097
#  0.0234 0.0298 0.0154 0.0034]