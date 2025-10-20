import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

#1. 데이터

docs = [
    '너무 재미있다.', '참 최고에요', '참 잘만든 영화에요.',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요.', '생각보다 지루해요', '연기가 어색해요',
    '재미 없어요.', '너무 재미없다.', '참 재밌네요.',
    '재현이 바보', '준희 잘생겼다.', '이삭이 또 구라친다.'
]

label = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,1,0])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
# {'참': 1, '너무': 2, '재미있다': 3, '최고에요': 4, '잘만든': 5, 
# '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, 
# '한': 10, '번': 11, '더': 12, '보고': 13, '싶어요': 14, '글쎄': 15, 
# '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, 
# '어색해요': 20, '재미': 21, '없어요': 22, '재미없다': 23, 
# '재밌네요': 24, '재현이': 25, '바보': 26, '준희': 27, 
# '잘생겼다': 28, '이삭이': 29, '또': 30, '구라친다': 31}
x = token.texts_to_sequences(docs)
# print(x)
# [[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], 
# [15], [16], [17, 18], [19, 20], [21, 22], [2, 23], 
# [1, 24], [25, 26], [27, 28], [29, 30, 31]]

from keras.preprocessing.sequence import pad_sequences
padding_x = pad_sequences(x, 
                          padding='pre',    #'post' default = 'pre'
                          maxlen=5,
                          truncating='pre', #'post' default = 'pre'
                         )

# print(padding_x)
# [[ 0  0  0  2  3]
#  [ 0  0  0  1  4]
#  [ 0  0  1  5  6]
#  [ 0  0  7  8  9]
#  [10 11 12 13 14]
#  [ 0  0  0  0 15]
#  [ 0  0  0  0 16]
#  [ 0  0  0 17 18]
#  [ 0  0  0 19 20]
#  [ 0  0  0 21 22]
#  [ 0  0  0  2 23]
#  [ 0  0  0  1 24]
#  [ 0  0  0 25 26]
#  [ 0  0  0 27 28]
#  [ 0  0 29 30 31]]
# print(padding_x.shape)  #(15, 5)

text = ['이삭이 참 잘생겼다.']
test_pred = token.texts_to_sequences(text)

# print(test_pred)    #[[29, 1, 28]]

padding_pred = pad_sequences(
    test_pred, padding='pre', maxlen=5, truncating='pre'
)

# print(padding_pred)
# [[0 0 0 0 0]
#  [0 0 0 0 0]
#  [0 0 0 0 0]
#  [0 0 0 0 0]
#  [0 0 0 0 1]
#  [0 0 0 0 0]
#  [0 0 0 0 0]
#  [0 0 0 0 0]
#  [0 0 0 0 0]
#  [0 0 0 0 0]
#  [0 0 0 0 0]]

x_train, x_test, y_train, y_test = train_test_split(
    padding_x, label
)

x_train = x_train.reshape(-1,5,1)
x_test = x_test.reshape(-1,5,1)
padding_pred = padding_pred.reshape(-1,5,1)

model = Sequential()
model.add(LSTM(256, activation='relu', input_shape=(5,1)))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

loss = model.evaluate(x_test, y_test)
results = model.predict(padding_pred)

print(results)
results=np.round(results)

if results==1:
    print('맞아요~')
else :
    print('아니에요~')

# [[1.]]
# 맞아요~



