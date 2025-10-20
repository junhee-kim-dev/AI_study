import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd

text = '오늘도 못생기고 영어를 디게 못하는 일삭이는 재미없는 \
    개그를 마구 마구 마구하면서 딴짓을 한다.'

token = Tokenizer()
token.fit_on_texts([text])

# print(token.word_index)
# {'마구': 1, '오늘도': 2, '못생기고': 3, '영어를': 4, '디게': 5, 
# '못하는': 6, '일삭이는': 7, '재미없는': 8, '개그를': 9, '마구하면서': 10, 
# '딴짓을': 11, '한다': 12}

# print(token.word_counts)
# OrderedDict([('오늘도', 1), ('못생기고', 1), ('영어를', 1), ('디게', 1), 
# ('못하는', 1), ('일삭이는', 1), ('재미없는', 1), ('개그를', 1), 
# ('마구', 2), ('마구하면서', 1), ('딴짓을', 1), ('한다', 1)])

x = token.texts_to_sequences([text])[0]
# print(x)    #[[2, 3, 4, 5, 6, 7, 8, 9, 1, 1, 10, 11, 12]]

from sklearn.preprocessing import OneHotEncoder

#1. pandas
# x = pd.get_dummies(x, dtype=int)

# #2. OneHotEncoder
# x = np.array(x).reshape(-1,1)
# ohe = OneHotEncoder(sparse=False)
# x = ohe.fit_transform(x)

# #3. keras
from keras.utils import to_categorical
x = to_categorical(x)

print(x)
print(x.shape)



