# https://www.kaggle.com/competitions/santander-customer-transaction-prediction

# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

print(x.shape)  #(200000, 200)
print(y.shape)  #(200000,)

# y = pd.get_dummies(y)
# ohe = OneHotEncoder(sparse=True)
# y = ohe.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123, stratify=y
)

ms = MaxAbsScaler()
ms.fit(x_train)
x_train = ms.transform(x_train)
x_test = ms.transform(x_test)
test_csv = ms.transform(test_csv)

# model = Sequential()
# model.add(Dense(1024, input_dim=200, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))

# input1 = Input(shape=(200,))
# dense1 = Dense(1024)(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(512)(drop1)
# drop2 = Dropout(0.4)(dense2)
# dense3 = Dense(256)(drop2)
# drop3 = Dropout(0.3)(dense3)
# dense4 = Dense(128)(drop3)
# output = Dense(1)(dense4)
# model = Model(inputs=input1, outputs=output)



# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(
#     monitor='val_loss', mode='auto',
#     patience=50, restore_best_weights=True
# )

# path1 = './_save/keras30/santander/'
# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
# filepath = ''.join([path1, 'San_', date, '_', filename])

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto',
#     save_best_only=True, filepath=filepath
# )
# s_time = time.time()
# hist = model.fit(
#     x_train, y_train, epochs=10000, batch_size=256, verbose=2,
#     validation_split=0.2, callbacks=[es,mcp]
# )
# e_time = time.time()

# loss = model.evaluate(x_test, y_test)
# results = model.predict(x_test)
# results = np.round(results)
# f1 = f1_score(y_test, results)

# y_submit = model.predict(test_csv)
# sub_csv['target'] = y_submit
# filename1 = ''.join(['submission_', date,'.csv'])
# sub_csv.to_csv(path+ filename1)
# print('File :', filename1)
# import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')
# # print(gpus) # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# if gpus:
#     print('GPU 있다~')
# else:
#     print('GPU 없다~')
# print('F1  :', f1)
# print('time:', np.round(e_time - s_time, 2), 'sec')


# File : submission_0608_2242.csv
# GPU 없다~
# F1  : 0.362078599366735
# time: 259.94 sec


# File : submission_0608_2248.csv
# GPU 있다~
# F1  : 0.4329590488771466
# time: 220.57 sec

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
models = [LinearSVC(C=0.3), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
for model in models :
    model.fit(x_train, y_train, )
    results = model.score(x_test, y_test)
    print(f'{model} :', results)
# LogisticRegression() : 0.914
# DecisionTreeClassifier() : 0.835425