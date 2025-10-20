import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler,RobustScaler,StandardScaler
from keras.callbacks import Callback
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
            
def RMSE(a, b) :
    return np.sqrt(mean_squared_error(a, b))
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
    x, y, train_size=0.9, test_size=0.1, shuffle=True, random_state=304)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


from keras.optimizers import Adam, Adagrad, SGD

#2. 모델 구성
model = Sequential()
model.add(Dense(40, input_dim=8))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    verbose=1,
    restore_best_weights=True,
)
rlr = ReduceLROnPlateau(
    monitor='val_loss', mode='min',
    patience=6, verbose=1, factor=0.5
)

hist = model.fit(
        x_train, y_train, epochs=100000, 
        batch_size=32, verbose=2, validation_split=0.2,
        callbacks=[es, rlr],
        )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
rmse = RMSE(y_test, results)
r2 = r2_score(y_test, results)
# 결과 저장
print(f"R2 {r2:.4f} | RMSE {rmse:.4f}")
