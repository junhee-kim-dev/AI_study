from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img       # 이미지 불러오기
from tensorflow.keras.preprocessing.image import img_to_array   # 불러온 이미지를 수치화
import matplotlib.pyplot as plt
import numpy as np

path = 'c:/Study25/_data/image/me/'
img = load_img(path + 'me.jpg', target_size=(100,100), )

# print(img)  #<PIL.Image.Image image mode=RGB size=100x100 at 0x173308879D0>
print(type(img))    #<class 'PIL.Image.Image'>
# PIL : Python Image Library

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
# print(arr)
print(arr.shape)    #(100, 100, 3)
# print(type(arr))    #<class 'numpy.ndarray'>

# arr.reshape(1,100,100,3)

img = np.expand_dims(arr, axis=0)   #(100, 100, 3)  # 이건 차원 늘리는거 밖에 못함
print(img.shape)                    #(1, 100, 100, 3)

# me폴더에 npy로 저장
np.save(path + 'keras47_me.npy', arr = img)