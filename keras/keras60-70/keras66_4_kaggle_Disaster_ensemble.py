# https://www.kaggle.com/competitions/nlp-getting-started/submissions

from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, SimpleRNN, Input, Flatten, concatenate
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import re

def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)  # URL 제거
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)   # 특수문자 제거
    text = text.lower()                          # 소문자 변환
    return text

path = './Study25/_data/kaggle/disaster/'

train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)
sub = pd.read_csv(path + 'sample_submission.csv')

def merge_text(df) :
    df['keyword'] = df['keyword'].fillna('unknown')
    df['location'] = df['location'].fillna('unknown').str.lower()
    df['text'] = df['text'] + ' keyword: ' + df['keyword'] + ' location: ' + df['location']
    return df

train_mer = merge_text(train)
test_mer = merge_text(test)
train_mer['text'] = train_mer['text'].apply(clean_text)
test_mer['text'] = test_mer['text'].apply(clean_text)

y = train['target']

train_1 = train_mer['keyword']
train_2 = train_mer['location']
train_3 = train_mer['text']
test_1 = test_mer['keyword']
test_2 = test_mer['location']
test_3 = test_mer['text']
# print(train_mer.columns)      #Index(['keyword', 'location', 'text', 'target'], dtype='object')

# print(train_mer['location'].value_counts())    # [7613 rows x 4 columns]
# unknown                         2535
# usa                              104
# new york                          75
# united states                     50
# london                            49
#                                 ... 
# todaysbigstock.com                 1
# buenos aires argentina             1
# everydaynigerian@gmail.com         1
# surulere lagos,home of swagg       1
# wilbraham, ma                      1
# Name: location, Length: 3233, dtype: int64
# print(test_mer)     # [3263 rows x 3 columns]
# exit()

# print(train_1.shape)    #(7613,)
# print(train_2.shape)    #(7613,)
# print(train_3.shape)    #(7613,)

oe_keyword = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
oe_location = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train_1 = oe_keyword.fit_transform(train_1.to_numpy().reshape(-1, 1))
test_1 = oe_keyword.transform(test_1.to_numpy().reshape(-1, 1))

train_2 = oe_location.fit_transform(train_2.to_numpy().reshape(-1, 1))
test_2 = oe_location.transform(test_2.to_numpy().reshape(-1, 1))

ohe_1 = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_2 = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
train_1 = ohe_1.fit_transform(train_1)
test_1 = ohe_1.transform(test_1)
train_2 = ohe_2.fit_transform(train_2)
test_2 = ohe_2.transform(test_2)

token = Tokenizer(num_words=10000)
token.fit_on_texts(list(train_3) + list(test_3))
train_3 = token.texts_to_sequences(train_3)
test_3 = token.texts_to_sequences(test_3)
train_3 = pad_sequences(train_3, maxlen=30)
test_3 = pad_sequences(test_3, maxlen=30)

train_1_x_train, train_1_x_test, train_2_x_train, train_2_x_test, train_3_x_train, train_3_x_test, y_train, y_test = train_test_split(
    train_1, train_2, train_3, y, random_state=42
)

# print(train_1.shape)    #(7613, 222)
# print(train_2.shape)    #(7613, 3233)
# print(train_3.shape)    #(7613, 30)
# print(test_1.shape)     #(3263, 222)
# print(test_2.shape)     #(3263, 3233)
# print(test_3.shape)     #(3263, 30)

input1 = Input(shape=(222,))
dense11 = Dense(512, activation='relu')(input1)
dense12 = Dense(256, activation='relu')(dense11)
dense13 = Dense(128, activation='relu')(dense12)

input2 = Input(shape=(3233,))
dense21 = Dense(512, activation='relu')(input2)
dense22 = Dense(256, activation='relu')(dense21)
dense23 = Dense(128, activation='relu')(dense22)

input3 = Input(shape=(30,))
embed = Embedding(input_dim=10000, output_dim=128, input_length=30)(input3)
rnn31 = SimpleRNN(128, activation='relu')(embed)
dense32 = Dense(128, activation='relu')(rnn31)
dense33 = Dense(128, activation='relu')(dense32)

merge = concatenate([dense13, dense23, dense33])
dense1 = Dense(128, activation='relu')(merge)
output = Dense(1, activation='sigmoid')(dense1)

model = Model(inputs=[input1, input2, input3], outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics='acc')
# model.summary()

es = EarlyStopping(
    monitor='val_loss', mode='auto', restore_best_weights=True, patience=15, verbose=2,
)

mcp = ModelCheckpoint(
    monitor='val_loss', mode='auto', save_best_only='True', verbose=2,
    filepath='./Study25/_data/kaggle/disaster/mcp.hdf5'
)

model.fit([train_1_x_train,train_2_x_train,train_3_x_train], y_train, epochs=100,
          verbose=2, validation_split=0.2, batch_size=128, callbacks=[es,mcp])

loss = model.evaluate([train_1_x_test,train_2_x_test,train_3_x_test], y_test)

results = model.predict([test_1, test_2, test_3])
sub['target'] = (results > 0.5).astype(int)
sub.to_csv(path+'sub/ensemble_submission.csv', index=False)

print('Loss :', loss[0])
print('ACC  :', loss[1])