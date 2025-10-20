from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import datetime



np_path = 'c:/Study25/_data/_save_npy/keras44/'
x = np.load(np_path + 'keras44_01_x.npy')
y = np.load(np_path + 'keras44_01_y.npy')
test = np.load(np_path + 'keras44_01_test.npy')
print('로드 됨.')

print(x.shape)  #(25000, 100, 100, 3)
print(y.shape)  #(25000,)

# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42,
)

print('가보자.')
model = Sequential()
model.add(Conv2D(32, 2, input_shape=(100,100,3), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(64, 3, activation='relu'))
# model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(32, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
es = EarlyStopping(
    monitor='val_acc', mode='max', verbose=1,
    patience=50, restore_best_weights=True,
)

path1 = './_data/kaggle/CatDog/mcp/'
date = datetime.datetime.now()
date = date.strftime('%H%M')
name = '({epoch:04d}_{val_loss:.4f}).hdf5'
f_path = ''.join([path1, 'k45_', date, '_', name])

mcp = ModelCheckpoint(
    monitor='val_acc', mode='max', verbose=1,
    save_best_only=True, filepath=f_path
)

s_time = time.time()
model.fit(
    x_train, y_train, epochs=100000, batch_size=128,
    verbose=2, validation_split=0.2, callbacks=[es, mcp]
)
e_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
results_round = np.round(results)
acc = accuracy_score(y_test, results_round)

print('CatDog')
print('Loss :', np.round(loss[0],4))
print('Acc  :', np.round(acc,4))
print('time :', np.round(e_time - s_time, 2), 'sec')

y_submit = model.predict(test)
submit_csv = pd.read_csv('./_data/kaggle/CatDog/sample_submission.csv')
submit_csv['label'] = y_submit
submit_csv.to_csv('./_data/kaggle/CatDog/submission_files/submission_files.csv', index=False)

# CatDog
# Loss : 0.5582
# Acc  : 0.7416
# time : 315.93 sec
