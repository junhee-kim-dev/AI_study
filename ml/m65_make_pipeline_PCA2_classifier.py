# 실습 06~11 을 한파일에 한번...! (fetch_cov 제외)

##############
#1. 데이터셋 (5개)
#2. 스케일러 (4개)
#3. 모델     (5개)

# [결과] : 6개의 데이터 셋에서 각...
#         어떤 모델, 어떤 스케일러에서 점수 어캐나오는지

# [실습준비]
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

x_06, y_06 = load_breast_cancer(return_X_y=True)
x_09, y_09 = load_wine(return_X_y=True)
x_11, y_11 = load_digits(return_X_y=True)

path_dbt = './Study25/_data/dacon/diabetes/'
path_bnk = './Study25/_data/kaggle/bank/'

trn_dbt = pd.read_csv(path_dbt + "train.csv", index_col=0)
trn_bnk = pd.read_csv(path_bnk + "train.csv", index_col=0)

x_07 = trn_dbt.drop(['Outcome'], axis=1)
y_07 = trn_dbt['Outcome']

x_08 = trn_bnk.drop(['Exited'], axis=1)
y_08 = trn_bnk['Exited']

x_07 = pd.get_dummies(x_07, drop_first=True)
x_08 = pd.get_dummies(x_08, drop_first=True)

best_scor_06 = 0
best_sclr_06 = ''
best_modl_06 = ''

best_scor_07 = 0
best_sclr_07 = ''
best_modl_07 = ''

best_scor_08 = 0
best_sclr_08 = ''
best_modl_08 = ''

best_scor_09 = 0
best_sclr_09 = ''
best_modl_09 = ''

best_scor_11 = 0
best_sclr_11 = ''
best_modl_11 = ''

list_data = [
     ('cancer', x_06, y_06,
      best_scor_06,
      best_sclr_06,
      best_modl_06,),
     ('diabetes', x_07, y_07,
      best_scor_07,
      best_sclr_07,
      best_modl_07,),
     ('bank', x_08, y_08,
      best_scor_08,
      best_sclr_08,
      best_modl_08,),
     ('wine', x_09, y_09,
      best_scor_09,
      best_sclr_09,
      best_modl_09,),
     ('digits', x_11, y_11,
      best_scor_11,
      best_sclr_11,
      best_modl_11,)]

list_sclr = [
     ('MinMaxScaler', MinMaxScaler()),
     ('MaxAbsScaler', MaxAbsScaler()),
     ('RobustScaler', RobustScaler()),
     ('StandardScaler', StandardScaler())
]

list_modl = [
     ('XGBClassifier', XGBClassifier(verbose=0)),
    #  ('LGBMClassifier', LGBMClassifier(verbosity=-1)),
     ('CatBoostClassifier', CatBoostClassifier(verbose=0)),
     ('RandomForestClassifier', RandomForestClassifier(verbose=0)),
     ('GradientBoostingClassifier', GradientBoostingClassifier(verbose=0))
]

for data, x, y, bscre, bsclr, bmodl in list_data:
     print('data :', data)
     x_trn, x_tst, y_trn, y_tst = train_test_split(
          x, y,
          train_size=0.7,
          shuffle=True,
          random_state=777,
          stratify=y
     )
     
     pca = PCA(n_components=8) 
     x = pca.fit_transform(x)
     
     for snme, sclr in list_sclr:
          for mnme, modl in list_modl:
               print('pipeline :', snme, mnme)
               model = make_pipeline(sclr, modl)
               
               model.fit(x_trn, y_trn)
               
               scr = model.score(x_tst, y_tst)
     
     if scr > bscre:
          bscre = scr
          bsclr = sclr
          bmodl = modl

print('06 :', best_scor_06, best_sclr_06, best_modl_06)
print('07 :', best_scor_07, best_sclr_07, best_modl_07)
print('08 :', best_scor_08, best_sclr_08, best_modl_08)
print('09 :', best_scor_09, best_sclr_09, best_modl_09)
print('11 :', best_scor_11, best_sclr_11, best_modl_11)