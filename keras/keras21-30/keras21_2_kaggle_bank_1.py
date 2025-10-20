# https://www.kaggle.com/competitions/playground-series-s4e1/submissions

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

path = './_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# train_csv = pd.read_csv(path + 'train.csv', index_col=[0,1,2])
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# test_csv = pd.read_csv(path + 'test.csv', index_col=[0,1,2])
submission_csv = pd.read_csv(path + 'sample_submission.csv')

# print(train_csv.shape)  #(165034, 11)
# print(test_csv.shape)   #(110023, 10)

# print(train_csv.head())
# print(train_csv.tail())
# print(train_csv.head(10))
# print(train_csv.isna().sum())
# print(test_csv.isna().sum())

train_csv[['Tenure', 'Balance']] = train_csv[['Tenure', 'Balance']].replace(0, np.nan)
train_csv = train_csv.fillna(train_csv.mean())
# print(train_csv)
# train_csv.to_csv(path + 'hello.csv')
test_csv[['Tenure', 'Balance']] = test_csv[['Tenure', 'Balance']].replace(0, np.nan)
test_csv = test_csv.fillna(test_csv.mean())

# 문자 데이터 수치화!!
# le = LabelEncoder()
# train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
oe = OrdinalEncoder()       # 이렇게 정의 하는 것을 인스턴스화 한다고 함
oe.fit(train_csv[['Geography', 'Gender']])
train_csv[['Geography', 'Gender']] = oe.transform(train_csv[['Geography', 'Gender']])
test_csv[['Geography', 'Gender']] = oe.transform(test_csv[['Geography', 'Gender']])

train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)

# print(test_csv.shape)   #(110023, 10)

# exit()
x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']
# print(x.shape)  #(165034, 10)
# print(y.shape)  #(165034,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

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

model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss', mode='min',
    patience=100, restore_best_weights=True,
)

rlrop = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, 
    patience=100
)

start_time = time.time()
hist = model.fit(
    x_train, y_train, epochs=100000, batch_size=1024,
    verbose=2, validation_split=0.2,
    callbacks=[es, rlrop]
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

y_submit = model.predict(test_csv)
y_submit = np.round(y_submit)
submission_csv['Exited'] = y_submit
submission_csv.to_csv(path + 'submission_0527_1.csv', index=False)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', label='loss')
plt.plot(hist.history['val_loss'], c='blue', label='val_loss')
plt.title('은행')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.grid()

plt.figure(figsize=(9,6))
plt.plot(hist.history['acc'], c='red', label='acc')
plt.plot(hist.history['val_acc'], c='blue', label='val_acc')
plt.title('은행')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# submission_0527.csv
# Loss : 0.3274894654750824
# Acc  : 0.8626351985942375
# F1   : 0.6347671983244724

# submission_0527_1.csv
# Loss : 0.3269093930721283
# Acc  : 0.862483715575484
# F1   : 0.6324993927617197