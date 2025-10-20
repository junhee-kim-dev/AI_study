# https://www.kaggle.com/competitions/playground-series-s4e1/submissions

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import f1_score
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = './_data/kaggle/bank/'
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
    x, y, train_size=0.8, shuffle=True, random_state=123
)


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(100, input_dim=10, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(70, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

input1= Input(shape=(10,))
dense1= Dense(100)(input1)
drop1 = Dropout(0.3)(dense1)
dense2= Dense(70)(drop1)
drop2 = Dropout(0.2)(dense2)
dense3= Dense(30)(drop2)
drop3 = Dropout(0.1)(dense3)
output1= Dense(1)(drop3)
model = Model(inputs= input1, outputs= output1)



model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss', mode='min',
    patience=100, restore_best_weights=True,
)


start_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=100000, batch_size=1024,
    verbose=2, validation_split=0.2,
    callbacks=[es, ]
)
end_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
results = np.round(results)
acc = accuracy_score(y_test, results)
f1 = f1_score(y_test, results)

print('Loss :', loss[0])
print('Acc  :', acc)
print('F1   :', f1)

# y_submit = model.predict(test_csv)
# y_submit = np.round(y_submit)
# submission_csv['Exited'] = y_submit
# submission_csv.to_csv(path + 'submission_0527_1.csv', index=False)


# Loss : 0.3281714916229248
# Acc  : 0.862483715575484
# F1   : 0.6295601077287195

# MinMaxScaler
# RMSE : 4.909639973231036
# R2 : 0.7542331069473764

# MaxAbsScaler
# RMSE : 5.063856531819747
# R2 : 0.7385510723780746

# StandardScaler
# RMSE : 5.049883344454853
# R2 : 0.7399919833098837

# RobustScaler
# RMSE : 4.89211886567054
# R2 : 0.7559840963384721