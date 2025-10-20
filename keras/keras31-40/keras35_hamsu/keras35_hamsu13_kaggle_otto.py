from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
import numpy as np
import pandas as pd
import datetime
import time

path = './_data/kaggle/otto/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
sub_csv = pd.read_csv(path + 'sampleSubmission.csv')

# print(train_csv.shape) #(61878, 94)
# print(test_csv.shape)  #(144368, 93)

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']

y = pd.get_dummies(y)
# print(y.shape)  #(61878, 9)
# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=123
)

mas = MaxAbsScaler()
mas.fit(x_train)
x_train = mas.transform(x_train)
x_test = mas.transform(x_test)
test_csv = mas.transform(test_csv)

model = Sequential()
model.add(Dense(300, input_dim=93, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(400, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(300, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(9, activation='softmax'))

input1 = Input(shape=(93,))
dense1 = Dense(300)(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(500)(drop1)
drop2 = Dropout(0.4)(dense2)
dense3 = Dense(400)(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(300)(drop3)
dense5 = Dense(200)(dense4)
output = Dense(9)(dense5)
model = Model(inputs=input1, outputs= output)







model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

es = EarlyStopping(
    monitor='val_loss', mode='min', patience=50,
    restore_best_weights=True
)

path1 = './_save/keras30/otto/'
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%m')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([path1, 'Otto_', date, '_', filename])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='min',
    save_best_only=True, filepath=filepath
)

s_time = time.time()
model.fit(
    x_train, y_train, epochs=100000, batch_size=256, 
    verbose=2, validation_split=0.2, callbacks=[es, mcp]
)
e_time = time.time()

loss= model.evaluate(x_test, y_test)
results = model.predict(x_test)
results = np.round(results)
f1 = f1_score(y_test, results, average='macro')

y_submit = model.predict(test_csv)
y_submit = np.argmax(y_submit, axis=1)
sub_csv['target'] = y_submit
filename1 = ''.join(['submission_', date, '.csv'])
sub_csv.to_csv(path + filename1)
print('File: ',filename1)

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
# print(gpus) # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
if gpus:
    print('GPU 있다~')
else:
    print('GPU 없다~')

print('F1  :', f1)
print('time:', np.round(e_time - s_time, 1), 'sec')


# File:  submission_0608_2206.csv
# GPU 없다~
# F1  : 0.769061762965092
# time: 130.2 sec

# File:  submission_0608_2206.csv
# GPU 있다~
# F1  : 0.7721535100421776
# time: 208.4 sec