from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import random
import numpy as np

seed = 123
random.seed(seed)
np.random.seed(seed)

datasets = load_iris()

x = datasets.data
y = datasets.target

import pandas as pd
from xgboost import XGBClassifier

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model4 = XGBClassifier(random_state=seed)

model4 = XGBClassifier(random_state=seed)
model4.fit(x_train, y_train)
print(f'{model4.__class__.__name__}')
print(f'ACC : {model4.score(x_test, y_test)}')                      # ACC : 0.9333333333333333
print(model4.feature_importances_)                                  # [0.02430454 0.02472077 0.7376847  0.21328996]

print('25%지점 :', np.percentile(model4.feature_importances_, 25))  # 25%지점 : 0.02461671084165573
per = np.percentile(model4.feature_importances_, 25)

col_names = []
# 삭제할 컬럼(25% 이하) 찾기
for i, fi in enumerate(model4.feature_importances_) :
    # print(i, fi)
    if fi <= per :
        col_names.append(datasets.feature_names[i])
    else :
        continue

x = pd.DataFrame(x, columns=datasets.feature_names)
x = x.drop(columns=col_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model4.fit(x_train, y_train)
print(f'ACC2 : {model4.score(x_test, y_test)}')      # ACC2 : 0.9333333333333333




