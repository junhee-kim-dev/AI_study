import numpy as np

path = './_data/_save_npy/keras46/gender/'
x = np.load(path + '(100,100)_flip_x.npy')
y = np.load(path + '(100,100)_flip_y.npy')

print(x.shape)      #(3309, 100, 100, 3)
print(y.shape)      #(3309,)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
optimizer=Adam(learning_rate=0.0001)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=190, stratify=y
)

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

from sklearn.utils import class_weight
cw = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
cw = dict(enumerate(cw))


es = EarlyStopping(monitor='val_acc', mode='max', verbose=1,
                   patience=30, restore_best_weights=True)
rl = ReduceLROnPlateau(monitor='val_acc', mode='max', verbose=1,
                       patience=10, factor=0.5)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='acc')
model.fit(x_train, y_train, epochs=10000, batch_size=64, verbose=2,
          validation_split=0.2,callbacks=[es], class_weight=cw)

loss = model.evaluate(x_test, y_test)
results = model.predict(x_test)
results_round = np.round(results)
f1 = f1_score(y_test, results_round)

print('### gender ###')
print('Loss :', np.round(loss[0], 4))
print('Acc  :', np.round(loss[1], 4))
print('F1   :', np.round(f1, 4))

# ### gender ###
# Loss : 0.7869
# Acc  : 0.6412
# F1   : 0.6877