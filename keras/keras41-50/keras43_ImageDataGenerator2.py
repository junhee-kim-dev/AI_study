from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_datagen = ImageDataGenerator(
    rescale=1/255.,                 # 가져오자마자 scaling, 정규화
    # horizontal_flip=True,           # 수평 뒤집기 -> 데이터 증폭 or 변환
    # vertical_flip=True,             # 수직 뒤집기 -> 데이터 증폭 or 변환
    # width_shift_range=0.1,          # 수치를 평행이동 10% -> 데이터 증폭 or 변환
    # height_shift_range=0.1,         # 수치를 수직이동 10%
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,                # 찌부 시키기
    # fill_mode='nearest',                      
)

test_datagen = ImageDataGenerator(  # test 데이터는 증폭하지 않는다.
    rescale=1/255.
)

path = './_data/image/brain/'
path_train = ''.join([path, 'train/'])
path_test = ''.join([path, 'test/'])

xy_train = train_datagen.flow_from_directory(
    path_train,                     # (160,150,150,1)
    target_size=(200, 200),         # (160,200,200,1)사이즈가 자동으로 조절됨      # Q. 확장되나? 축소되나? 100,100 짜리면 남은 공간이 0이 되나?
    batch_size=160,                  # (10, 200,200,1) *16 세트가 됨     데이터 수를 batchsize로 나눔
    class_mode='binary',            # 이진 분류
    color_mode='grayscale',
    shuffle=True,
    seed=42
)

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200, 200),
    batch_size=160,
    class_mode='binary',
    color_mode='grayscale',
    # shuffle=True,                   # default = False
    # seed=42
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt

# plt.imshow(x_train[1], 'gray')
# plt.show()
# exit()

def model_S():
    model = Sequential()
    model.add(Conv2D(32, 2, input_shape=(200,200,1), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 2, activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, 2, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
    return model

model = model_S()
es = EarlyStopping(
    monitor='val_loss', mode='min', verbose=1,
    restore_best_weights=True, patience=40
)

path = './_save/keras43/brain/'
date = datetime.datetime.now()
date = date.strftime('%H%M')
name = '({epoch:04d}-{val_loss:.4f}).hdf5'
f_path = ''.join([path, 'k43_', date, '_', name])

mcp = ModelCheckpoint(
    monitor='val_loss', mode='min', verbose=1,
    save_best_only=True, filepath=f_path
)

s_time = time.time()
model.fit(x_train, y_train, epochs=100000, batch_size=64,
          verbose=1, validation_split=0.2, callbacks=[es,mcp])
e_time = time.time()

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
results_round = np.round(results)
acc = accuracy_score(y_test, results_round)
print('K43_IDG')
print('ACC :', np.round(acc,4))
print('Loss:', np.round(loss[0],4))

# K43_IDG
# ACC : 0.6917
# Loss: 0.6775

# K43_IDG
# ACC : 0.9833
# Loss: 0.059

# K43_IDG
# ACC : 0.9917
# Loss: 0.0549

# K43_IDG
# ACC : 1.0
# Loss: 0.0201