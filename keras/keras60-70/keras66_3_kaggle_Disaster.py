from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import re

# ================= 데이터 ==================
def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

path = './Study25/_data/kaggle/disaster/'
train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
sub = pd.read_csv(path + 'sample_submission.csv')

def merge_text(df):
    df['keyword'] = df['keyword'].fillna('unknown')
    df['location'] = df['location'].fillna('unknown').str.lower()
    df['text'] = df['text'] + ' keyword: ' + df['keyword'] + ' location: ' + df['location']
    return df

train = merge_text(train)
test = merge_text(test)

train['text'] = train['text'].apply(clean_text)
test['text'] = test['text'].apply(clean_text)

# ================= 텍스트 전처리 ==================
token = Tokenizer(num_words=10000)
token.fit_on_texts(pd.concat([train['text'], test['text']]))

X = token.texts_to_sequences(train['text'])
X_test = token.texts_to_sequences(test['text'])

X = pad_sequences(X, maxlen=30)
X_test = pad_sequences(X_test, maxlen=30)

y = train['target'].values

# ================= 모델 ==================
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=30))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X, y, epochs=100, batch_size=128, validation_split=0.2, callbacks=[es])

# ================= 예측 및 제출 ==================
pred = (model.predict(X_test) > 0.5).astype(int)
sub['target'] = pred
sub.to_csv(path + 'sub/final_submission.csv', index=False)
