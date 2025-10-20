from sklearn.datasets import load_digits

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import time, datetime
import matplotlib.pyplot as plt

# 시드 고정
seed = 123
import random
random.seed(seed)
np.random.seed(seed)
# tf.random.set_seed(seed)

datasets = load_digits()
x = datasets.data
y = datasets.target

# print(x)
# print(x.shape)  #(1797, 64)
# print(y)
# print(y.shape)  #(1797,)
# print(x[0])
# aaa= x[1].reshape(8,8)
# print(aaa)
# print(y[1])
# exit()
# print(datasets.DESCR)
# print(datasets.feature_names)

# print(np.max(x), np.min(x))
# print(np.max(y), np.min(y))
# # x_show = x.reshape(-1,8,8,1)
# # import random
# # n = random.randint(1,200)
# # print(n)
# # print(y[n])
# # plt.imshow(x_show[n], 'gray')
# # # plt.show()

# # exit()

# y = pd.get_dummies(y)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, train_size=0.8, shuffle=True,
# )

# ms = MaxAbsScaler()
# ms.fit(x_train)
# x_train = ms.transform(x_train)
# x_test = ms.transform(x_test)

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)
# # exit()
# x_train = x_train.reshape(-1,8,8,1)
# x_test = x_test.reshape(-1,8,8,1)
# from tensorflow.keras.layers import Dropout, Flatten, Conv2D
# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# import time

# model = Sequential()
# model.add(Conv2D(64, (2,2), strides=1, input_shape=(8,8,1), padding='same'))
# model.add(Conv2D(64, (2,2), padding='same'))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (2,2),activation='relu', padding='same'))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# #3. 컴파일, 훈련

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss', mode='min',
#     patience=50, restore_best_weights=True, verbose=1
# )

# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# path1 = './_save/keras41/11digits/'
# filename = '({epoch:04d}-{val_loss:.4f}).hdf5'
# filepath = ''.join([path1, 'k41_', date, '_', filename])

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='min',
#     save_best_only=True, filepath=filepath,
#     verbose=1
# )

# s_time = time.time()
# hist = model.fit(
#     x_train, y_train, epochs=10000, batch_size=64,
#     verbose=2, validation_split=0.2,
#     callbacks=[es, mcp]
# )
# e_time = time.time()
# loss = model.evaluate(x_test, y_test)
# results = model.predict(x_test)

# y_test_arg = np.argmax(y_test, axis=1)
# results_arg = np.argmax(results, axis=1)
# acc = accuracy_score(y_test_arg, results_arg)
# f1 = f1_score(y_test_arg, results_arg, average='macro')
# print('CNN 11')
# print('Loss :', loss[0])
# print('acc :', acc)
# print('time :', np.round(e_time - s_time, 1), 'sec')


# print('Acc  :', acc)
# print('F1   :', f1)
# images = datasets.images
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 2))  # 가로 길게
# for i in range(10):
#     plt.subplot(1, 10, i + 1)
#     plt.imshow(images[i], cmap='gray')  # 흑백 이미지
#     plt.axis('off')                     # 축 제거
#     plt.title(str(y[i]), fontsize=10)

# plt.tight_layout()
# plt.show()

# from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
# n_split=5
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=333)
# model = HistGradientBoostingClassifier()
# scores = cross_val_score(model, x, y, cv=kfold)         # 훈련 평가가 합쳐진 형태
# print('acc :', scores, '\n평균 acc :', round(np.mean(scores),4))

# acc : [0.98055556 0.98333333 0.96657382 0.96935933 0.97493036] 
# 평균 acc : 0.975

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=seed, train_size=0.75, stratify=y
)
print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([133, 136, 133, 137, 136, 136, 136, 134, 131, 135]))


################## SMOTE 적용 ####################
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=seed,
              k_neighbors=5,           #default
            #   sampling_strategy='auto' #default
            #   sampling_strategy=0.75   # 최대값의 75% 수준
            #   sampling_strategy={0:5000, 1:5000, 2:5000},
            #   n_jobs=-1   # 0.13버전에서 삭제
              sampling_strategy={0:5000, 
                                 1:5000, 
                                 2:5000,
                                 3:5000,
                                 4:5000, 
                                 5:5000, 
                                 6:5000,
                                 7:5000, 
                                 8:5000, 
                                 9:5000},            
              )

x_train, y_train = smote.fit_resample(x_train, y_train)
print(np.unique(y_train,return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([137, 137, 137, 137, 137, 137, 137, 137, 137, 137]))
print(x_train.shape)    #(1370, 64)
# exit()

ss = StandardScaler()
ss.fit(x_train)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

from xgboost import XGBClassifier
#2. 모델구성
model = XGBClassifier(random_state=seed)

model.fit(x_train, y_train)
results = model.predict(x_test)
acc = accuracy_score(y_test, results)
f1 = f1_score(y_test, results, average='macro')

print(acc)
print(f1)

#         acc : [0.97912317 0.96868476 0.95824635]
# average acc : 0.96868
#    test acc : 0.9305555555555556

# 0.9688888888888889
# 0.9691922599537705

# 0.9711111111111111
# 0.9713642644501871