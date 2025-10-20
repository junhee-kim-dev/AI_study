from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten, Conv2D, BatchNormalization, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#1.data
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape)    #(60000, 28, 28)
# print(y_train.shape)    #(60000,)
# print(x_test.shape)     #(10000, 28, 28)
# print(y_test.shape)     #(10000,)
# print(type(x_train))    #<class 'numpy.ndarray'>

x_train = x_train.reshape(60000,28*28*1)
x_test = x_test.reshape(10000,28*28*1)

ohe = OneHotEncoder(sparse=False)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

x_train = x_train/255.
x_test = x_test/255.

# print(y_train.shape)    #(60000, 10)
# print(y_test.shape)

model = Sequential()
# model.add(Conv2D(64, (2,2), input_shape=(28,28,1),
#                  padding='same', activation='relu'))
# model.add(MaxPooling2D())
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2), activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Conv2D(16, (2,2), activation='relu'))
# # model.summary() #(-1,11,11,32)
# model.add(Flatten())
model.add(Dense(128, input_dim=784, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1,
    patience=50, restore_best_weights=True)

path = './_save/keras40/fashion_mnist/DNN_GPU_'
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

print('fashion_mnist')
print('Loss :', np.round(loss[0], 4))
print('Acc  :', np.round(acc, 4))
print('time :', np.round(e_time - s_time, 1), 'sec')


# fashion_mnist
# CNN_CPU              


# CNN_GPU
# Loss : 0.2245
# Acc  : 0.9255
# time : 200.2 sec

# DNN_CPU
# Loss : 0.3244
# Acc  : 0.8834
# time : 58.3 sec

# DNN_GPU
# Loss : 0.317
# Acc  : 0.887
# time : 226.4 sec