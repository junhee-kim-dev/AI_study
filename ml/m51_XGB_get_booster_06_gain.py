from math import gamma
import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
dataset = load_breast_cancer()

x = dataset.data
y = dataset.target
print(x.shape)

le = LabelEncoder()
y = le.fit_transform(y)

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import EarlyStopping

seed =123

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

ss_1 = StandardScaler()
x_train = ss_1.fit_transform(x_train)
x_test = ss_1.transform(x_test)

model4 = XGBClassifier(
        gamma        = 0,
        subsample    = 0.4,
        reg_alpha    = 0,
        reg_lambda   = 1,
        max_depth    = 6,
        n_estimators = 10000,
        eval_metric='logloss',
        random_state = seed,
        early_stopping_rounds = 20,
)

model4.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose = 0)
# print('########### ')
# print(f'#   기존 R2 : {model4.score(x_test, y_test)}') 

score_dict = model4.get_booster().get_score(importance_type='gain')
print(score_dict)
total = sum(score_dict.values())
print(total)

score_list = [score_dict.get(f"f{i}", 0)/total for i in range(x.shape[1])]
# f"f{i}" 의 첫 f 는 formatted string (에프스트링)
# 
print(score_list)
print(len(score_list))

# exit()

# key = list(aaa.keys())
# print(key)
# value = list(aaa.values())
# print(value)
# max = np.max(value)
# print(max)
# value_max = value / max
# print(value_max)

# [0.01663922 0.01457612 0.00670366 0.0121365  0.00946518 0.00998936
#  0.01549685 0.33480835 0.01313937 0.00612979 0.0131958  0.00340235
#  0.00496585 0.02094395 0.00557439 0.01210066 0.00898831 0.00324526
#  0.00789787 0.00842577 0.05203583 0.01735101 0.1999523  0.01903242
#  0.01266684 0.00241838 0.03777545 0.11876774 0.00774672 0.00442872]

thresholds = np.sort(score_list)
print(thresholds)

# [0.         0.00142329 0.00167072 0.00213313 0.00238045 0.00281381
#  0.00295481 0.0048802  0.00509214 0.00659898 0.00679698 0.00761755
#  0.00867345 0.00888047 0.01064892 0.01151786 0.01291873 0.01790295
#  0.01876996 0.01883145 0.01979863 0.01985464 0.02007599 0.0272005
#  0.03191127 0.03899821 0.04105973 0.13189764 0.20633852 0.31035901]

# exit()

from sklearn.feature_selection import SelectFromModel
for i in thresholds :
    selection = SelectFromModel(model4, threshold=i, prefit=False)
    # threshold 가 i값 이상인 것을 모두 훈련시킨다.
    # prefit = False : 모델이 아직 학습되지 않았을 때, fit을 호출해서 훈련한다.(default)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    if select_x_train.shape[1] == 0:
        continue    
    # print(select_x_train.shape)
    
    select_model = XGBClassifier(
        gamma        = 0,
        subsample    = 0.4,
        reg_alpha    = 0,
        reg_lambda   = 1,
        max_depth    = 6,
        n_estimators = 10000,
        eval_metric='logloss',
        random_state = seed,
        early_stopping_rounds = 20,
        )
    
    select_model.fit(select_x_train, y_train, eval_set =[(select_x_test, y_test)], verbose=0)
    
    select_y_pred = select_model.predict(select_x_test)
    # print(select_y_pred)
    acc = accuracy_score(y_test, select_y_pred)
    print(f'Treshold:{i:.4f}, Columns:{select_x_train.shape[1]}, ACC:{acc*100:.4f}%')

# Treshold:0.0024, Columns:29, ACC:99.1228%
# Treshold:0.0032, Columns:28, ACC:100.0000%
# Treshold:0.0034, Columns:27, ACC:100.0000%
# Treshold:0.0044, Columns:27, ACC:100.0000%
# Treshold:0.0050, Columns:26, ACC:100.0000%
# Treshold:0.0056, Columns:24, ACC:99.1228%
# Treshold:0.0061, Columns:23, ACC:99.1228%
# Treshold:0.0067, Columns:22, ACC:99.1228%
# Treshold:0.0077, Columns:21, ACC:99.1228%
# Treshold:0.0079, Columns:20, ACC:99.1228%
# Treshold:0.0084, Columns:20, ACC:99.1228%
# Treshold:0.0090, Columns:18, ACC:99.1228%
# Treshold:0.0095, Columns:17, ACC:100.0000%
# Treshold:0.0100, Columns:16, ACC:99.1228%
# Treshold:0.0121, Columns:16, ACC:99.1228%
# Treshold:0.0121, Columns:14, ACC:100.0000%
# Treshold:0.0127, Columns:13, ACC:99.1228%
# Treshold:0.0131, Columns:12, ACC:98.2456%
# Treshold:0.0132, Columns:11, ACC:98.2456%
# Treshold:0.0146, Columns:10, ACC:98.2456%
# Treshold:0.0155, Columns:9, ACC:99.1228%
# Treshold:0.0166, Columns:8, ACC:99.1228%
# Treshold:0.0174, Columns:7, ACC:96.4912%
# Treshold:0.0190, Columns:6, ACC:96.4912%
# Treshold:0.0209, Columns:6, ACC:96.4912%
# Treshold:0.0378, Columns:4, ACC:96.4912%
# Treshold:0.0520, Columns:3, ACC:96.4912%
# Treshold:0.1188, Columns:2, ACC:96.4912%
# Treshold:0.2000, Columns:1, ACC:92.9825%
exit()
per = np.percentile(value, 25)

col_names = []
# 삭제할 컬럼(25% 이하) 찾기
for i, fi in enumerate(value) :
    # print(i, fi)
    if fi <= per :
        col_names.append(dataset.feature_names[i])
    else :
        continue

print(col_names)
# ['mean perimeter', 'mean fractal dimension', 'texture error', 'perimeter error', 
# 'smoothness error', 'concave points error', 'worst compactness', 'worst fractal dimension']

x_f = pd.DataFrame(x, columns=dataset.feature_names)
x1 = x_f.drop(columns=col_names)
x2 = x_f[col_names]
print(x1.shape)
print(x2.shape)

x1_train, x1_test, x2_train, x2_test= train_test_split(
    x1, x2, train_size=0.8, random_state=seed,
)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x2_train = pca.fit_transform(x2_train)
x2_test = pca.transform(x2_test)

x3_train = np.concatenate([x1_train, x2_train], axis=1)
x3_test = np.concatenate([x1_test, x2_test], axis=1)
print(x3_train.shape)

# exit()

model4.fit(x3_train, y_train)
score = model4.score(x3_test, y_test)
print(f'# ACC : {model4.score(x3_test, y_test)}') 
