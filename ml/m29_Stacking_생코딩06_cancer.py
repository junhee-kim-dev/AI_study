import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor,LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

r = np.random.randint(1, 1000)

X, Y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.9, random_state=r
)

mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)

# 2-1 모델
xgb = XGBClassifier()
# rf = RandomForestRegressor()
cat = CatBoostClassifier(verbose=0)
lg = LGBMClassifier(verbosity=-1)

models = [xgb,cat,lg]

train_list = []
test_list = []

for model in models:
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_list.append(y_train_pred)
    test_list.append(y_test_pred)
    
    score = accuracy_score(y_test, y_test_pred)
    class_name = model.__class__.__name__
    print('{0} R2 : {1:.4f}'.format(class_name, score))
    
# XGBRegressor R2 : 0.8374
# RandomForestRegressor R2 : 0.8079
# CatBoostRegressor R2 : 0.8559
# LGBMRegressor R2 : 0.8335

x_train_new = np.array(train_list).T
x_test_new = np.array(test_list).T

print(x_train_new.shape)    #(18576, 4)

#2-2 모델

model2 = CatBoostRegressor(verbose=0)
model2.fit(x_train_new, y_train)
y_pred2 = model2.predict(x_test_new)

y_pred22 = model2.predict(x_test_new)
score2 = r2_score(y_test, y_pred2)
print(score2)

# 0.8409059653552557

