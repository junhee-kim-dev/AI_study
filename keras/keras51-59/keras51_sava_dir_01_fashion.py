from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

(x_train,y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train/255.
x_test = x_test/255.

datagen = ImageDataGenerator(
    rescale=1/255., vertical_flip=False, horizontal_flip=True, width_shift_range=0.1,
    height_shift_range=0, rotation_range=15, zoom_range=1, shear_range=0.8, fill_mode='nearest' 
)

augment_size = 40000
rand_idx = np.random.randint(x_train.shape[0], size=augment_size)

# print(rand_idx) #[56310  4062  9700 ... 11436 10713 37510]
# print(np.min(rand_idx), np.max(rand_idx))   #0 59997

aug_x = x_train[rand_idx].copy()    # 4만개의 데이터를 copy, copy로 새로운 메모리 할당.
                                    # 서로 영향이 없어짐
# print(aug_x)
# print(aug_x.shape)      #(40000, 28, 28)
aug_y = y_train[rand_idx].copy()
# print(aug_y.shape)      #(40000,)
# xy_data = datagen.flow(
#     x=, y=,
#     batch_size=32, shuffle=False
# )

# aug_x = aug_x.reshape(-1, 28,28,1)
aug_x = aug_x.reshape(
    aug_x.shape[0],
    aug_x.shape[1],
    aug_x.shape[1], 1)
# print(aug_x.shape)      # (40000, 28, 28, 1)
path = 'c:/STUDY25/_data/_save_img/01_fashion/'
aug_x = datagen.flow(
    aug_x, aug_y, batch_size=40000, shuffle=False,
    save_to_dir=path
).next()[0]
exit()
# print(aug_x.shape)  #(40000, 28, 28, 1)

# print(x_train.shape)    #(60000, 28, 28)
x_train = x_train.reshape(-1,28,28,1)
# print(x_train.shape)    #(60000, 28, 28, 1)
x_test = x_test.reshape(-1,28,28,1)
# print(x_test.shape)     # (10000, 28, 28, 1)

x_train = np.concatenate((x_train, aug_x))
y_train = np.concatenate((y_train, aug_y))
# print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)

# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# exit()

#############맹그러봐#############

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28,28,1),
                 padding='same', activation='relu'))
model.add(MaxPooling2D())
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Conv2D(16, (2,2), activation='relu'))
# model.summary() #(-1,11,11,32)
model.add(Flatten())
model.add(Dense(32, activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1,
    patience=50, restore_best_weights=True)

path = './_save/keras50/fashion_mnist/'
name = '({epoch:04d}-{val_acc:.4f}).hdf5'
file = ''.join([path, name])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', verbose=1,
    save_best_only=True, filepath = file
)

s_time = time.time()
model.fit(
    x_train, y_train, epochs=10000, batch_size=128,
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
# Loss : 0.2361
# Acc  : 0.9221
# time : 243.2 sec

# Loss : 0.2262
# Acc  : 0.9173
# time : 176.6 sec

# Loss : 0.2245
# Acc  : 0.9255
# time : 200.2 sec

# Loss : 0.2315
# Acc  : 0.9229
# time : 240.0 sec

# aug_fashion_mnist
# Loss : 0.5684
# Acc  : 0.9008
# time : 284.1 sec










