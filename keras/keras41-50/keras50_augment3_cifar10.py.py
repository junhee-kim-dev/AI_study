from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train/255.
x_test = x_test/255.

# print(x_train.shape)    #(50000, 32, 32, 3)
# print(x_test.shape)     #(10000, 32, 32, 3)
# print(y_train.shape)    #(50000, 1)
# print(y_test.shape)     #(10000, 1)

datagen = ImageDataGenerator(
    rescale=1/255.,
    horizontal_flip=True, width_shift_range=0.1, rotation_range=20,
    zoom_range=1.2, fill_mode='nearest'
)

augment_size = 50000

aug_x = x_train.copy()
aug_y = y_train.copy()

aug_x = aug_x.reshape(-1,32,32,3)
aug_x = datagen.flow(
    aug_x, aug_y, batch_size=50000, shuffle=False
).next()[0]

x_train = np.concatenate((x_train, aug_x))
y_train = np.concatenate((y_train, aug_y))

print(x_train.shape)
print(np.min(x_train), np.max(x_train))
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)


model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(32,32,3), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (2,2), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(
    loss='categorical_crossentropy', optimizer='adam',metrics='acc'
)

es = EarlyStopping(
    monitor='val_acc', mode='auto', verbose=1,
    restore_best_weights=True, patience=50
)

path = './_save/keras50/cifar10/'
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

print('cifar10')
print('loss :', np.round(loss[0], 4))
print('acc  :', np.round(acc,4))
print('time :', np.round(e_time - s_time, 1), 'sec')

# cifar10

# loss : 0.802
# acc  : 0.7379
# time : 204.6 sec

# loss : 0.8115
# acc  : 0.7403
# time : 209.6 sec

# loss : 0.5642
# acc  : 0.8305
# time : 513.5 sec

# loss : 0.5507
# acc  : 0.8349
# time : 655.0 sec


# cifar10
# loss : 0.9934
# acc  : 0.6624
# time : 191.7 sec


