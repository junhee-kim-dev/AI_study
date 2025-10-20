from tensorflow.keras.datasets import cifar100

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape)    #(50000, 32, 32, 3)
# print(x_test.shape)     #(10000, 32, 32, 3)
# print(y_train.shape)    #(50000, 1)
# print(y_test.shape)     #(10000, 1)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

# print(y_train.shape)    #(50000, 100)
# print(y_test.shape)     #(10000, 100)

x_train = x_train/255.
x_test = x_test/255.

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(32,32,3), activation='relu'))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

input = Input(shape=(32,32,3))
conv1 = Conv2D(64, 3, activation='relu')(input)
maxp1 = MaxPooling2D()(conv1)
batc1 = BatchNormalization()(maxp1)
drop1 = Dropout(0.2)(batc1)
conv2 = Conv2D(128, 2, activation='relu')(drop1)
maxp2 = MaxPooling2D()(conv2)
batc2 = BatchNormalization()(maxp2)
conv3 = Conv2D(64, 3, activation='relu')(batc2)
flatt = Flatten()(conv3)
dens1 = Dense(256, activation='relu')(flatt)
batc3 = BatchNormalization()(dens1)
drop2 = Dropout(0.2)(batc3)
dens2 = Dense(128, activation='relu')(drop2)
outpu = Dense(100, activation='softmax')(dens2)
model = Model(inputs=input, outputs=outpu)


model.compile(
    loss='categorical_crossentropy', optimizer='adam',metrics='acc'
)

es = EarlyStopping(
    monitor='val_acc', mode='auto', verbose=1,
    patience=50, restore_best_weights=True,
)

path = './_save/keras39/cifar100/'
name = '({epoch:04d}-{val_acc:.4f}).hdf5'
file = ''.join([path, name])

mcp = ModelCheckpoint(
    monitor='val_acc', mode='auto', verbose=1,
    save_best_only=True, filepath = file
)

s_time = time.time()
model.fit(
    x_train, y_train, epochs=1000000, batch_size=128,
    verbose=2, validation_split=0.1, callbacks=[es, mcp]
)
e_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
y_test_arg = np.argmax(y_test, axis=1)
results_arg = np.argmax(results, axis=1)
acc = accuracy_score(y_test_arg, results_arg)

print('cifar100')
print('loss :', np.round(loss[0],4))
print('acc  :', np.round(acc, 4))
print('time :', np.round(e_time - s_time, 4))

# cifar100