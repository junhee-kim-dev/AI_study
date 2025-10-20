import numpy as np
import pandas as pd
import sklearn as sk

from sklearn.datasets import load_wine
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target
# print(type(x))
# print(type(datasets))
print(len(datasets.feature_names))  # 13
# print(len(x.feature_names)) 
# exit()

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score

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
        n_estimators = 100,
        eval_metric='mlogloss',
        random_state = seed,
    )

model4.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose = 0)

score_dict = model4.get_booster().get_score(importance_type='gain')
print(score_dict)
score_list = np.array([score_dict.get(f"f{i}", 0) for i in range(x.shape[1])])
print(score_list.tolist())
# exit()
total = np.sum(score_list)
score_list_2 = score_list/total

print(len(score_list_2))    # 12
# exit()
feature_names = list(datasets.feature_names)
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
    
    if select_x_train.shape[1] == 0:
        print(f"⚠️  Threshold {i:.4f}에서 선택된 피처가 없습니다. 건너뜁니다.\n")
        continue
    
    select_model = XGBClassifier(
        gamma        = 0,
        subsample    = 0.4,
        reg_alpha    = 0,
        reg_lambda   = 1,
        max_depth    = 6,
        n_estimators = 100,
        eval_metric='mlogloss',
        random_state = seed,
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

if len(delete_columns)==0 :
    print('삭제할 컬럼이 없습니다!')    
else :    
    print(f'삭제할 컬럼은 다음과 같습니다. \n{delete_columns}')

# 삭제할 컬럼이 없습니다!