from sklearn.datasets import load_digits

# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import time, datetime

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

# y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True,
)

ms = MaxAbsScaler()
ms.fit(x_train)
x_train = ms.transform(x_train)
x_test = ms.transform(x_test)

# from tensorflow.keras.layers import Dropout


# model = Sequential()
# model.add(Dense(256, input_dim=64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# input1 = Input(shape=(64,))
# dense1 = Dense(256)(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(256)(drop1)
# drop2 = Dropout(0.4)(dense2)
# dense3 = Dense(100)(drop2)
# drop3 = Dropout(0.3)(dense3)
# dense4 = Dense(100)(drop3)
# output = Dense(10)(dense4)
# model = Model(inputs= input1, outputs = output)


# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# es = EarlyStopping(
#     monitor='val_loss', mode='auto',
#     patience=100,
#     restore_best_weights=True
# )

# path = './_save/keras30/digits/'
# date = datetime.datetime.now()
# date = date.strftime('%m%d_%H%M')
# filename = '{epoch:04d}_{val_loss:.4f}.hdf5'
# filepath = ''.join([path, 'digits_', date, '_', filename])

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode='auto',
#     save_best_only=True,
#     filepath=filepath
# )

# s_time = time.time()
# hist = model.fit(x_train, y_train, epochs=10000000, 
#           batch_size=256, verbose=3,
#           validation_split=0.2, callbacks=[es, mcp])
# e_time = time.time()

# loss = model.evaluate(x_test, y_test)
# results = model.predict(x_test)
# results = np.round(results)
# # results = np.argmax(results, axis=1)
# # acc = accuracy_score(y_test, results)
# # f1 = f1_score(y_test, results)
# np.set_printoptions(threshold=np.inf)
# print('Loss :', loss[0])
# print(results)
# # print('Acc  :', acc)
# # print('F1   :', f1)
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


from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
models = [LinearSVC(C=0.3), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
for model in models :
    model.fit(x_train, y_train)
    results = model.score(x_test, y_test)
    print(f'{model} :', results)

# LinearSVC(C=0.3) : 0.9555555555555556
# LogisticRegression() : 0.9638888888888889
# DecisionTreeClassifier() : 0.8305555555555556
# RandomForestClassifier() : 0.9694444444444444