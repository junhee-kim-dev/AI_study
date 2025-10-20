import numpy as np

path = './_data/_save_npy/keras46/rps/'
x = np.load(path + '(100,100)x.npy')
y = np.load(path + '(100,100)y.npy')

print(x.shape)      #(2048, 100, 100, 3)
print(y.shape)      #(2048, 3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd

print(y.shape)  #(2048, 3)
# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=42
)

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(100,100,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2,
          validation_split=0.2,)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
y_test_arg = np.argmax(y_test, axis=1)
results_arg = np.argmax(results, axis=1)
f1 = f1_score(y_test_arg, results_arg, average='macro')

print('### rps ###')
print('Loss :', np.round(loss[0], 4))
print('Acc  :', np.round(loss[1], 4))
print('F1   :', np.round(f1, 4))


# ### rps ###
# Loss : 0.0003
# Acc  : 1.0
# F1   : 1.0