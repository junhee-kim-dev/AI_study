import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score
import warnings
import pandas as pd
from sklearn.utils import all_estimators

warnings.filterwarnings('ignore')

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

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=333, test_size=0.2,
)

from sklearn.model_selection import KFold, StratifiedKFold
n_split = 3
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)

#2. 모델구성
# model = RandomForestRegression()
allAlgorithm = all_estimators(type_filter='classifier')

# print('모델의 갯수 :' , len(allAlgorithm))  #모델의 갯수 : 55
# print(type(allAlgorithm))   <class 'list'>

for (name, algorithm) in allAlgorithm:

    try :
        print(f'############# {name} ###############')
        model = algorithm()
        if name in ['KNeighborsClassifier', 'SVC', 'NuSVC']:
            continue
        #3. 컴파일 훈련
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        
        #4. 평가 예측
        model.fit(x_train, y_train)
        results = model.predict(x_test)
        results = np.round(results)
        acc = accuracy_score(y_test, results)
        print(f'{name}의 cross_val_score   정답률:')
        print(np.round(np.mean(score),5))
        print(f'{name}의 cross_val_predict 정답률:')
        print(np.round(acc,5))

    except :
        print('오류 발생')