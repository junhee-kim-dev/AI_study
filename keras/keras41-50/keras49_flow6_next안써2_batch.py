from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

augment_size = 100                    # 증가시킬 사이즈

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

aaa = np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1)
# print(aaa.shape)        # (78400,)      -> (100, 28, 28, 1)

datagen = ImageDataGenerator(
    rescale=1/255.,                   # 가져오자마자 scaling, 정규화
    horizontal_flip=True,             # 수평 뒤집기 -> 데이터 증폭 or 변환
    # vertical_flip=True,             # 수직 뒤집기 -> 데이터 증폭 or 변환
    width_shift_range=0.1,            # 수치를 평행이동 10% -> 데이터 증폭 or 변환
    # height_shift_range=0.1,         # 수치를 수직이동 10%
    rotation_range=15,
    # zoom_range=1.2,
    # shear_range=0.7,                # 찌부 시키기
    fill_mode='nearest',                      
)

xy_data = datagen.flow(x=aaa,     # x데이터
    y=np.zeros(augment_size),     # y데이터
    batch_size=32,
    shuffle=False,) #.next()      # next는 딸깍임 그냥 이름이 next여서 그렇지 그냥 딸깍임.

# print(xy_data)  #<keras.preprocessing.image.NumpyArrayIterator object at 0x000001E1EF92C460>
print(type(xy_data))              # <class 'keras.preprocessing.image.NumpyArrayIterator'>
print(len(xy_data))               # 4

print(xy_data[0][0].shape)        # (32, 28, 28, 1)
print(xy_data[0][1].shape)        # (32,)

print(xy_data[3][0].shape)        # (4, 28, 28, 1)
print(xy_data[3][1].shape)        # (4,)


# exit()
# plt.figure(figsize=(10,10))
# for i in range(100) :
#     plt.subplot(10,10, i+1)
#     plt.imshow(xy_data[0][0][i], cmap='gray')   # .next() 빼면 차원하나 늘려야함

# plt.show()













