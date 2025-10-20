import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

from sklearn.utils import all_estimators
# import sklearn as sk
# print(sk.__version__)   #1.2.2

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=333, test_size=0.2,
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.model_selection import KFold, StratifiedKFold
n_split = 3
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)

#2. 모델구성
# model = RandomForestRegression()
allAlgorithm = all_estimators(type_filter='regressor')

# print('모델의 갯수 :' , len(allAlgorithm))  #모델의 갯수 : 55
# print(type(allAlgorithm))   <class 'list'>


for (name, algorithm) in allAlgorithm:
    try :
        model = algorithm()
        
        #3. 컴파일 훈련
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        
        #4. 평가 예측
        results = cross_val_predict(model, x_test, y_test, cv=kfold)
        r2 = r2_score(y_test, results)
        print('################################')
        print(f'{name}의 cross_val_score   정답률:')
        print(np.round(np.mean(score),5))
        print(f'{name}의 cross_val_predict 정답률:')
        print(np.round(r2,5))
        
    except :
        print('오류 발생')

# ################################
# ARDRegression의 cross_val_score   정답률:
# 0.52031
# ARDRegression의 cross_val_predict 정답률:
# 0.27493
# ################################
# AdaBoostRegressor의 cross_val_score   정답률:
# 0.46626
# AdaBoostRegressor의 cross_val_predict 정답률:
# 0.47069
# ################################
# BaggingRegressor의 cross_val_score   정답률:
# 0.7792
# BaggingRegressor의 cross_val_predict 정답률:
# 0.71623
# ################################
# BayesianRidge의 cross_val_score   정답률:
# 0.52098
# BayesianRidge의 cross_val_predict 정답률:
# 0.26128
# 오류 발생
# ################################
# DecisionTreeRegressor의 cross_val_score   정답률:
# 0.58474
# DecisionTreeRegressor의 cross_val_predict 정답률:
# 0.47325
# ################################
# DummyRegressor의 cross_val_score   정답률:
# -2e-05
# DummyRegressor의 cross_val_predict 정답률:
# -0.00012
# ################################
# ElasticNet의 cross_val_score   정답률:
# 0.14455
# ElasticNet의 cross_val_predict 정답률:
# 0.13323
# ################################
# ElasticNetCV의 cross_val_score   정답률:
# 0.51677
# ElasticNetCV의 cross_val_predict 정답률:
# 0.11662
# ################################
# ExtraTreeRegressor의 cross_val_score   정답률:
# 0.54286
# ExtraTreeRegressor의 cross_val_predict 정답률:
# 0.45016
# ################################
# ExtraTreesRegressor의 cross_val_score   정답률:
# 0.80326
# ExtraTreesRegressor의 cross_val_predict 정답률:
# 0.75307
# ################################
# GammaRegressor의 cross_val_score   정답률:
# -4.59428
# GammaRegressor의 cross_val_predict 정답률:
# 0.33121
# ################################
# GaussianProcessRegressor의 cross_val_score   정답률:
# -34.10576
# GaussianProcessRegressor의 cross_val_predict 정답률:
# -1.62058
# ################################
# GradientBoostingRegressor의 cross_val_score   정답률:
# 0.7882
# GradientBoostingRegressor의 cross_val_predict 정답률:
# 0.75937
# ################################
# HistGradientBoostingRegressor의 cross_val_score   정답률:
# 0.83205
# HistGradientBoostingRegressor의 cross_val_predict 정답률:
# 0.78721
# ################################
# HuberRegressor의 cross_val_score   정답률:
# -3.7509
# HuberRegressor의 cross_val_predict 정답률:
# -1.24096
# 오류 발생
# ################################
# KNeighborsRegressor의 cross_val_score   정답률:
# 0.68463
# KNeighborsRegressor의 cross_val_predict 정답률:
# 0.62968
# ################################
# KernelRidge의 cross_val_score   정답률:
# -1.12405
# KernelRidge의 cross_val_predict 정답률:
# -1.18145
# ################################
# Lars의 cross_val_score   정답률:
# 0.48744
# Lars의 cross_val_predict 정답률:
# 0.26215
# ################################
# LarsCV의 cross_val_score   정답률:
# 0.5409
# LarsCV의 cross_val_predict 정답률:
# 0.14692
# ################################
# Lasso의 cross_val_score   정답률:
# -2e-05
# Lasso의 cross_val_predict 정답률:
# -0.00012
# ################################
# LassoCV의 cross_val_score   정답률:
# 0.54343
# LassoCV의 cross_val_predict 정답률:
# 0.14817
# ################################
# LassoLars의 cross_val_score   정답률:
# -2e-05
# LassoLars의 cross_val_predict 정답률:
# -0.00012
# ################################
# LassoLarsCV의 cross_val_score   정답률:
# 0.5409
# LassoLarsCV의 cross_val_predict 정답률:
# 0.14692
# ################################
# LassoLarsIC의 cross_val_score   정답률:
# 0.52102
# LassoLarsIC의 cross_val_predict 정답률:
# 0.26215
# ################################
# LinearRegression의 cross_val_score   정답률:
# 0.52102
# LinearRegression의 cross_val_predict 정답률:
# 0.26215
# ################################
# LinearSVR의 cross_val_score   정답률:
# -3.46552
# LinearSVR의 cross_val_predict 정답률:
# -1.04862
# ################################
# MLPRegressor의 cross_val_score   정답률:
# 0.37521
# MLPRegressor의 cross_val_predict 정답률:
# 0.67029
# 오류 발생
# 오류 발생
# 오류 발생
# 오류 발생
# 오류 발생
# ################################
# NuSVR의 cross_val_score   정답률:
# 0.68654
# NuSVR의 cross_val_predict 정답률:
# 0.65871
# ################################
# OrthogonalMatchingPursuit의 cross_val_score   정답률:
# 0.47425
# OrthogonalMatchingPursuit의 cross_val_predict 정답률:
# 0.4681
# ################################
# OrthogonalMatchingPursuitCV의 cross_val_score   정답률:
# 0.49857
# OrthogonalMatchingPursuitCV의 cross_val_predict 정답률:
# 0.11569
# 오류 발생
# ################################
# PLSRegression의 cross_val_score   정답률:
# 0.40086
# PLSRegression의 cross_val_predict 정답률:
# 0.09639
# ################################
# PassiveAggressiveRegressor의 cross_val_score   정답률:
# -63.62329
# PassiveAggressiveRegressor의 cross_val_predict 정답률:
# -38.31351
# ################################
# PoissonRegressor의 cross_val_score   정답률:
# 0.38542
# PoissonRegressor의 cross_val_predict 정답률:
# 0.41225
# ################################
# QuantileRegressor의 cross_val_score   정답률:
# -0.05394
# QuantileRegressor의 cross_val_predict 정답률:
# -0.05824
# ################################
# RANSACRegressor의 cross_val_score   정답률:
# -3.04848
# RANSACRegressor의 cross_val_predict 정답률:
# -4.72182
# 오류 발생
# ################################
# RandomForestRegressor의 cross_val_score   정답률:
# 0.79994
# RandomForestRegressor의 cross_val_predict 정답률:
# 0.73961
# 오류 발생
# ################################
# Ridge의 cross_val_score   정답률:
# 0.52096
# Ridge의 cross_val_predict 정답률:
# 0.26085
# ################################
# RidgeCV의 cross_val_score   정답률:
# 0.52046
# RidgeCV의 cross_val_predict 정답률:
# 0.26202
# ################################
# SGDRegressor의 cross_val_score   정답률:
# -6.014457704731348e+20
# SGDRegressor의 cross_val_predict 정답률:
# -7.351537189811414e+19
# ################################
# SVR의 cross_val_score   정답률:
# 0.68507
# SVR의 cross_val_predict 정답률:
# 0.65493
# 오류 발생
# ################################
# TheilSenRegressor의 cross_val_score   정답률:
# -12.38681
# TheilSenRegressor의 cross_val_predict 정답률:
# -5.16948
# ################################
# TransformedTargetRegressor의 cross_val_score   정답률:
# 0.52102
# TransformedTargetRegressor의 cross_val_predict 정답률:
# 0.26215
# ################################
# TweedieRegressor의 cross_val_score   정답률:
# 0.23901
# TweedieRegressor의 cross_val_predict 정답률:
# 0.11121
# 오류 발생

