import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
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

#2. 모델구성
# model = RandomForestRegression()
allAlgorithm = all_estimators(type_filter='regressor')

# print('모델의 갯수 :' , len(allAlgorithm))  #모델의 갯수 : 55
# print(type(allAlgorithm))   <class 'list'>

max_score=0
max_name = 'default'

for (name, algorithm) in allAlgorithm:
    try :
        model = algorithm()
        
        #3. 컴파일 훈련
        model.fit(x_train, y_train)
        
        #4. 평가 예측
        score = model.score(x_test, y_test)
        
        print(name, '의 정답률:')
        print(score)
        
        if score > max_score:
            max_score = score
            max_name = name
            print('현재 최고 모델:', name)
        
    except :
        print('오류 발생')

print('=================================')
print('최고 모델 :', max_name, max_score)
print('=================================')

# ARDRegression 의 정답률:
# 0.6006525418956501
# 현재 최고 모델: ARDRegression
# AdaBoostRegressor 의 정답률:
# 0.42114272701238786
# BaggingRegressor 의 정답률:
# 0.7778275428341122
# 현재 최고 모델: BaggingRegressor
# BayesianRidge 의 정답률:
# 0.5998175665053438
# 오류 발생
# DecisionTreeRegressor 의 정답률:
# 0.6005223547667148
# DummyRegressor 의 정답률:
# -0.0009862822479156375
# ElasticNet 의 정답률:
# 0.1473845970904002
# ElasticNetCV 의 정답률:
# 0.5995065686590978
# ExtraTreeRegressor 의 정답률:
# 0.5479749533088232
# ExtraTreesRegressor 의 정답률:
# 0.8059452581605805
# 현재 최고 모델: ExtraTreesRegressor
# GammaRegressor 의 정답률:
# 0.30110789634220725
# GaussianProcessRegressor 의 정답률:
# -140.745501094275
# GradientBoostingRegressor 의 정답률:
# 0.7804548381607719
# HistGradientBoostingRegressor 의 정답률:
# 0.8273939037353999
# 현재 최고 모델: HistGradientBoostingRegressor
# HuberRegressor 의 정답률:
# 0.3484227470763708
# 오류 발생
# KNeighborsRegressor 의 정답률:
# 0.6741670815105287
# KernelRidge 의 정답률:
# -1.1905333415264159
# Lars 의 정답률:
# 0.5998241027086236
# LarsCV 의 정답률:
# 0.597455155893773
# Lasso 의 정답률:
# -0.0009862822479156375
# LassoCV 의 정답률:
# 0.5981984235659354
# LassoLars 의 정답률:
# -0.0009862822479156375
# LassoLarsCV 의 정답률:
# 0.597455155893773
# LassoLarsIC 의 정답률:
# 0.5998241027086236
# LinearRegression 의 정답률:
# 0.5998241027086236
# LinearSVR 의 정답률:
# -1.2434754985837109
# MLPRegressor 의 정답률:
# 0.7089113482739147
# 오류 발생
# 오류 발생
# 오류 발생
# 오류 발생
# 오류 발생
# NuSVR 의 정답률:
# 0.6757191550250314
# OrthogonalMatchingPursuit 의 정답률:
# 0.4693071917449224
# OrthogonalMatchingPursuitCV 의 정답률:
# 0.4693071917449224
# 오류 발생
# PLSRegression 의 정답률:
# 0.5151269335369386
# PassiveAggressiveRegressor 의 정답률:
# -34.907650664437526
# PoissonRegressor 의 정답률:
# 0.37988816036018447
# QuantileRegressor 의 정답률:
# -0.04305020483091049
# RANSACRegressor 의 정답률:
# 0.45477931074809286
# 오류 발생
# RandomForestRegressor 의 정답률:
# 0.8028022680278095
# 오류 발생
# Ridge 의 정답률:
# 0.5998142461470968
# RidgeCV 의 정답률:
# 0.5998231917158309
# SGDRegressor 의 정답률:
# -1.9750135476472842e+24
# SVR 의 정답률:
# 0.6749431678825287
# 오류 발생
# TheilSenRegressor 의 정답률:
# -6.682875786115525
# TransformedTargetRegressor 의 정답률:
# 0.5998241027086236
# TweedieRegressor 의 정답률:
# 0.34729806628715587
# 오류 발생
# =================================
# 최고 모델 : HistGradientBoostingRegressor 0.8273939037353999
# =================================
