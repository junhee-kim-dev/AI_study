# https://www.kaggle.com/competitions/playground-series-s4e1/submissions

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import f1_score
from keras.layers import BatchNormalization
from keras.layers import Dropout
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = './Study25/_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')

train_csv[['Tenure', 'Balance']] = train_csv[['Tenure', 'Balance']].replace(0, np.nan)
train_csv = train_csv.fillna(train_csv.mean())

test_csv[['Tenure', 'Balance']] = test_csv[['Tenure', 'Balance']].replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

oe = OrdinalEncoder()       # 이렇게 정의 하는 것을 인스턴스화 한다고 함
oe.fit(train_csv[['Geography', 'Gender']])
train_csv[['Geography', 'Gender']] = oe.transform(train_csv[['Geography', 'Gender']])
test_csv[['Geography', 'Gender']] = oe.transform(test_csv[['Geography', 'Gender']])

train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']



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
        eval_metric='logloss',
        random_state = seed,
        # early_stopping_rounds = 20,
    )

model4.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose = 0)

score_dict = model4.get_booster().get_score(importance_type='gain')

value = list(score_dict.values())
total = np.sum(value)
score_list_2 = value/total

feature_names = list(x.columns)
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
        n_estimators = 100,
        eval_metric='logloss',
        random_state = seed,
        # early_stopping_rounds = 20,
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

# ['CreditScore', 'Tenure', 'HasCrCard', 'EstimatedSalary']