from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(compo_1)  713
# print(compo_2)  486
# print(compo_3)  331
# print(compo_4)  154

x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

x_train = x_train/255.
x_test = x_test/255.

num = [154, 331, 486, 713, 784]

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import datetime, time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

for i in num :
    pca = PCA(n_components=i)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)
    
    model = Sequential()
    model.add(Dense(128, input_dim=i, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2, seed=42))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2, seed=42))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2, seed=42))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2, seed=42))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

    s_time = time.time()
    hist = model.fit(x_train_pca, y_train, epochs=50, 
            batch_size=256, verbose=2,
            validation_split=0.2,)
    e_time = time.time()

    loss = model.evaluate(x_test_pca, y_test)
    results = model.predict(x_test_pca)
    result_arg = np.argmax(results, axis=1)
    y_test_arg = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_test_arg, result_arg)

    # print('Acc :', np.round(acc, 4))
    print('time:', np.round(e_time - s_time, 1), 'sec')
    print(f'{i}의 값:', np.round(acc, 4))

# time: 28.4 sec
# 154의 값: 0.979

# time: 23.3 sec
# 331의 값: 0.9773

# time: 15.9 sec
# 486의 값: 0.9743

# time: 20.4 sec
# 713의 값: 0.9745

# time: 17.2 sec
# 784의 값: 0.975