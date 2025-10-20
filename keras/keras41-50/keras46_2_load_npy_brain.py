import numpy as np
import pandas as pd

npy_path = 'c:/Study25/_data/_save_npy/keras46/brain/'
x_train = np.load(npy_path + 'brain(200,200)_x_train.npy')
y_train = np.load(npy_path + 'brain(200,200)_y_train.npy')
x_test = np.load(npy_path + 'brain(200,200)_x_test.npy')
y_test = np.load(npy_path + 'brain(200,200)_y_test.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(200,200,1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=2,
          validation_split=0.2,)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
results_round = np.round(results)
f1 = f1_score(y_test, results_round)

print('### brain ###')
print('Loss :', np.round(loss[0], 4))
print('Acc  :', np.round(loss[1], 4))
print('F1   :', np.round(f1, 4))

# ### brain ###
# Loss : 0.0309
# Acc  : 0.975
# F1   : 0.9744