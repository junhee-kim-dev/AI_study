# 47 카피

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img       # 이미지 불러오기
from tensorflow.keras.preprocessing.image import img_to_array   # 불러온 이미지를 수치화
import matplotlib.pyplot as plt
import numpy as np

path = 'c:/Study25/_data/image/me/'
img = load_img(path + 'me.jpg', target_size=(300,300), )

# print(img)  #<PIL.Image.Image image mode=RGB size=100x100 at 0x173308879D0>
# print(type(img))    #<class 'PIL.Image.Image'>
# PIL : Python Image Library

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
# print(arr)
# print(arr.shape)    #(100, 100, 3)
# print(type(arr))    #<class 'numpy.ndarray'>

# arr.reshape(1,100,100,3)

img = np.expand_dims(arr, axis=0)   #(100, 100, 3)  # 이건 차원 늘리는거 밖에 못함
# print(img.shape)                    #(1, 100, 100, 3)

# me폴더에 npy로 저장
# np.save(path + 'keras47_me.npy', arr = img)

############### 여기부터 증폭 ###############


datagen = ImageDataGenerator(
    rescale=1/255.,                   # 가져오자마자 scaling, 정규화
    horizontal_flip=True,           # 수평 뒤집기 -> 데이터 증폭 or 변환
    vertical_flip=True,             # 수직 뒤집기 -> 데이터 증폭 or 변환
    width_shift_range=0.1,            # 수치를 평행이동 10% -> 데이터 증폭 or 변환
    height_shift_range=0.1,         # 수치를 수직이동 10%
    rotation_range=15,
    zoom_range=1.2,
    shear_range=0.7,                # 찌부 시키기
    fill_mode='nearest',                      
)

it = datagen.flow(img,              # flow 수치화된 이미지 데이터 불러오기
    batch_size=1,)


print('##################################')
print(it)   #<keras.preprocessing.image.NumpyArrayIterator object at 0x000002700A902160>
print('##################################')
# 파이썬 2.0 문법
# aaa = it.next()         
# print(aaa)
# print(aaa.shape)        #(1, 100, 100, 3)

# 지금 문법
# bbb = next(it)          
# print(bbb)
# print(bbb.shape)          #(1, 100, 100, 3)

# print(it.next())
# print(it.next())
# print(it.next())            # 원래는 총 데이터 수 이상 .next 쓰면 에러남

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(9,5))
for i in range(10):
    # batch = it.next()
    batch = next(it)
    print(batch.shape)      # (1,100,100,3)
    batch = batch.reshape(300,300,3)
    ax[i//5][i%5].imshow(batch)
    ax[i//5][i%5].axis('off')

plt.show()






