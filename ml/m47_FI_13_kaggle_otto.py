from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
import numpy as np
import pandas as pd
import datetime
import time

path = './Study25/_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')

# print(train_csv.shape) #(61878, 94)
# print(test_csv.shape)  #(144368, 93)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

from sklearn.preprocessing import LabelEncoder
la = LabelEncoder()
y = la.fit_transform(y)

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

seed = 72

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model4 = XGBClassifier(random_state=seed)
model4.fit(x_train, y_train)
print('13')
print(f'# {model4.__class__.__name__}')
print(f'# ACC : {model4.score(x_test, y_test)}') 
print('#', model4.feature_importances_)

print('# 25%지점 :', np.percentile(model4.feature_importances_, 25))
per = np.percentile(model4.feature_importances_, 25)

col_names = []
# 삭제할 컬럼(25% 이하) 찾기
for i, fi in enumerate(model4.feature_importances_) :
    # print(i, fi)
    if fi <= per :
        col_names.append(x.columns[i])
    else :
        continue

x = pd.DataFrame(x, columns=x.columns)
x = x.drop(columns=col_names)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model4.fit(x_train, y_train)
print(f'# ACC2 : {model4.score(x_test, y_test)}')

# XGBClassifier
# ACC : 0.8100355526826115
# [0.00452094 0.00608761 0.00445097 0.00319807 0.00355696 0.00265722
#  0.00930468 0.00754148 0.01302011 0.00286376 0.10247942 0.00171758
#  0.00376595 0.02386406 0.02422675 0.00499869 0.01133567 0.0028187
#  0.00799392 0.0059091  0.00375564 0.00237373 0.00755122 0.00509091
#  0.00950926 0.02672535 0.00345753 0.00271105 0.00528571 0.01637685
#  0.00281201 0.00503193 0.0036908  0.04316122 0.01027585 0.02439317
#  0.00367797 0.00460975 0.02587686 0.01514156 0.00990009 0.01752234
#  0.0092647  0.00250932 0.00623185 0.00245374 0.01647533 0.00541108
#  0.00264066 0.01032189 0.00356203 0.00398064 0.01669602 0.00382969
#  0.00382219 0.00755185 0.01159124 0.0078856  0.01397958 0.05524632
#  0.00364131 0.01033441 0.00268535 0.00598209 0.00201635 0.00331154
#  0.01077843 0.02503778 0.02756603 0.00323722 0.00563783 0.00735605
#  0.00262968 0.00217394 0.01971294 0.00585302 0.00801449 0.01387008
#  0.00541283 0.00485987 0.00408534 0.00221708 0.00458347 0.01058355
#  0.00461129 0.01195708 0.0037058  0.00580826 0.0032339  0.07247422
#  0.01454812 0.00786372 0.00552075]
# 25%지점 : 0.0036779672373086214
# ACC2 : 0.807207498383969