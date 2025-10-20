from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy

train_datagen = ImageDataGenerator(
    rescale=1/255.,                 # 가져오자마자 scaling, 정규화
    horizontal_flip=True,           # 수평 뒤집기 -> 데이터 증폭 or 변환
    vertical_flip=True,             # 수직 뒤집기 -> 데이터 증폭 or 변환
    width_shift_range=0.1,          # 수치를 평행이동 10% -> 데이터 증폭 or 변환
    height_shift_range=0.1,         # 수치를 수직이동 10%
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,                # 찌부 시키기
    fill_mode='nearest',                      
)

test_datagen = ImageDataGenerator(  # test 데이터는 증폭하지 않는다.
    rescale=1/255.
)

path = './_data/image/brain/'
path_train = ''.join([path, 'train/'])
path_test = ''.join([path, 'test/'])

xy_train = train_datagen.flow_from_directory(
    path_train,                     # (160,150,150,1)
    target_size=(200, 200),         # (160,200,200,1)사이즈가 자동으로 조절됨      # Q. 확장되나? 축소되나? 100,100 짜리면 남은 공간이 0이 되나?
    batch_size=10,                  # (10, 200,200,1) *16 세트가 됨     데이터 수를 batchsize로 나눔
    class_mode='binary',            # 이진 분류
    color_mode='grayscale',
    shuffle=True,
)
# Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(200, 200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    # shuffle=True,                   # default = False
)
# Found 120 images belonging to 2 classes.

# print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000248B4202100>
# print(xy_test)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000248EA277D30>
# print(xy_train[15])
# print(xy_test[0])
# print(len(xy_train))              # 16
# print(len(xy_test))               # 12
# print(xy_train[0][0].shape)       # (10, 200, 200, 1) # xy_train의 [0]
# print(xy_train[2][1].shape)       # (10,)
# print(xy_train[3][1].shape)       # (10,)
# print(xy_train[2][0].shape)       # (10, 200, 200, 1)
# print(xy_train[1][0].shape)       # (10, 200, 200, 1)
# print(xy_train[0].shape)          # 에러AttributeError: 'tuple' object has no attribute 'shape'
# print(xy_train[16])               # 에러ValueError: Asked to retrieve element 16, but the Sequence has length 16
# print(xy_train[0][2])             # 에러IndexError: tuple index out of range
# print(type(xy_train))             # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))          # <class 'tuple'>
# print(type(xy_train[0][0]))       # <class 'numpy.ndarray'>
# print(type(xy_train[0][1]))       # <class 'numpy.ndarray'>