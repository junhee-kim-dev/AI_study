# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import datetime
import time

path = './Study25/_data/kaggle/santander/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sample_submission.csv')

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

# print(x.shape)
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
        eval_metric='logloss',
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
    
# ['var_3', 'var_14', 'var_17', 'var_30', 'var_38', 'var_39', 'var_42', 'var_47', 'var_61', 'var_96', 'var_100', 'var_103', 'var_120', 'var_124', 'var_129', 'var_136', 'var_160', 'var_185']