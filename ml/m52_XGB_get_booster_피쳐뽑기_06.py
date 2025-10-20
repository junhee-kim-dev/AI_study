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

score_dict = model4.get_booster().get_score(importance_type='gain')

value = list(score_dict.values())
total = np.sum(value)
score_list_2 = value/total

feature_names = list(dataset.feature_names)
score_df = pd.DataFrame({
    'feature' : feature_names,  # [feature_names[int(f[1:])] for f in score_dict.keys()]
    'gain' : score_list_2       # list(score_dict.values())
}).sort_values(by='gain', ascending=True)

print(score_df, '\n')
thresholds = score_df['gain']

delete_columns = []
max_acc = 0

from sklearn.feature_selection import SelectFromModel
for idx, i in enumerate(thresholds) :
    print(f'================{idx+1}================')
    
    selection = SelectFromModel(model4, threshold=i, prefit=False)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
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
    
    deleted_col = np.array(feature_names)[~selection.get_support()]
    
    mask = selection.get_support()
    deleted = [feature_names[j] for j, selected in enumerate(mask) if not selected]
    
    
    select_y_pred = select_model.predict(select_x_test)
    acc = accuracy_score(y_test, select_y_pred)

    if acc > max_acc :
        max_acc = acc
        delete_columns = deleted
    
    print(f'Treshold:{i:.4f}, n:{select_x_train.shape[1]}, ACC:{acc*100:.4f}%')
    print(f'==================================\n')
    
    # print(f'삭제된 컬럼 : {deleted_col.tolist()}\n')
    # print(f'삭제된 컬럼 : {deleted}\n')

print(f'삭제할 컬럼은 다음과 같습니다. \n{delete_columns}')
