from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Conv1D, LSTM, Flatten
import numpy as np
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ================= 데이터 =================
docs = [
    '너무 재미있다.', '참 최고에요', '참 잘만든 영화에요.',
    '추천하고 싶은 영화입니다.', '한 번 더 보고 싶어요.', '글쎄',
    '별로에요.', '생각보다 지루해요', '연기가 어색해요',
    '재미 없어요.', '너무 재미없다.', '참 재밌네요.',
    '재현이 바보', '준희 잘생겼다.', '이삭이 또 구라친다.',
    '이삭이 시끄럽다.', '수업은 따라가자.', '시킨 것은 하고 놀자.',
    '너 잘났다.', '너가 알면 얼마나 알겠어?', '선생님 보다 잘해?',
    '작작해.', '입 닥쳐.', '헛소리 그만 해.', '니 똥 굵다.',
    '얼마나 안다고 그렇게 떠들어.', '그만 떠들어.', '니는 뭔데?',
    '헛소리 작작해.', '입이 방정이지.', '선생님 최고에요.', 
    '사실 아니에요.', '피곤해요.', '커피 주세요.', '밥 주세요.',
    '약 주세요.', '야미~', '입이 간질거려요.', '1등이다.'
]
label = np.array([1,1,1,1,1,0,0,0,0,0,0,1,0,
                  1,0,0,0,0,1,1,0,0,0,0,1,0,
                  1,0,0,0,1,0,0,1,1,1,1,0,0])

token = Tokenizer()
token.fit_on_texts(docs)
x = token.texts_to_sequences(docs)
word_size = len(token.word_index)
print(word_size)
exit()
x = pad_sequences(x, padding='pre', maxlen=5)

# ================= 원핫 인코딩 =================
x_ohe = to_categorical(x, num_classes=word_size + 1)

# ================= 테스트 데이터 =================
text = ['이삭이 참 잘생겼다.']
test_seq = token.texts_to_sequences(text)
test_seq = pad_sequences(test_seq, padding='pre', maxlen=5)
test_ohe = to_categorical(test_seq, num_classes=word_size + 1)

# ================= 데이터 분할 =================
x_train, x_test, y_train, y_test = train_test_split(x_ohe, label, random_state=42)

# print(x_train.shape)    #(29, 5, 74)
# print(x_test.shape)     #(10, 5, 74)
# print(test_ohe.shape)   #(1, 5, 74)
# exit()
# ================= 모델 =================
model = Sequential()
model.add(Conv1D(256, 3, activation='relu', input_shape=(5, word_size + 1)))
model.add(Conv1D(256, 3, activation='relu'))
model.add(LSTM(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

# ================= 평가 및 예측 =================
loss = model.evaluate(x_test, y_test)
results = model.predict(test_ohe)

print('예측 결과:', results)
results = np.round(results)

if results == 1:
    print('맞아요~')
else:
    print('아니에요~')

# 예측 결과: [[0.9999997]]
# 맞아요~