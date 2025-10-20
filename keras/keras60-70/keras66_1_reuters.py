import keras
from tensorflow.keras.utils import to_categorical
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(
    num_words=1000, # 단어 사전의 개수, 빈도수가 높은 단어 순으로 1000개
    test_split=0.2, # 
    maxlen=2376,    # 단어 길이가 100개까지 있는 문장
)

# print(x_train.shape)    #(8982,)
# print(x_test.shape)     #(2246,)
# print(y_train.shape)    #(8982,)
# print(y_test.shape)     #(2246,)

# print(y_train[0])       #3
# print(np.unique(y_train))

from keras.utils import pad_sequences
padding_x_train = pad_sequences(x_train, maxlen=500)
padding_x_test = pad_sequences(x_test, maxlen=500)
# padding_y_train = pad_sequences(y_train)
# padding_y_test = pad_sequences(y_test)

# print(padding_x_train.shape)    #(8982, 2376)
# print(padding_x_test.shape)     #(2246, 2376)

# max = max(len(i) for i in x_train)
# min = min(len(i) for i in x_train)
# sum = sum(map(len, x_train))/len(x_train)
# print(max)  #2246
# print(min)  #13
# print(sum)  #145.29150428682775

y_train_ca = to_categorical(y_train)
y_test_ca = to_categorical(y_test)
# print(y_train.shape)            #(8981, 46)
# print(y_test.shape)             #(2246, 46)


from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=256, input_length=500))
model.add(LSTM(256,))
model.add(Dense(128,))
model.add(Dense(46, activation='softmax'))

es = EarlyStopping(
    monitor='val_loss', mode='min', restore_best_weights=True, patience=10
)

path = './_save/keras66/mcp.hdf5'
mcp = ModelCheckpoint(
    monitor='val_loss', mode='min', save_best_only=True, filepath=path
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(padding_x_train, y_train_ca, epochs=100, batch_size=128, validation_split=0.2, callbacks=[es, mcp])

loss = model.evaluate(padding_x_test, y_test_ca)
# results = model.predict(padding_x_test)
# results_arg = np.argmax(results)

print('ACC  :', loss[1])
print('LOSS :', loss[0])
# print('예측값: ', results_arg)

# ACC  : 0.687889575958252
# LOSS : 1.3204246759414673
