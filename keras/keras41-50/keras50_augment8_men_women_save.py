from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

orginal_datagen = ImageDataGenerator(
    rescale=1/255.
)

path = './_data/kaggle/men_women/'
train = orginal_datagen.flow_from_directory(
    path, target_size=(100,100), batch_size=100,
    class_mode='binary', color_mode='rgb', shuffle=False
)

men_x = []
men_y = []

for i in range(len(train)):
    ox, oy = train[i]
    idx = np.where(oy==0)[0]
    men_x.append(ox[idx])
    men_y.append(oy[idx])

x_men = np.concatenate(men_x, axis=0)
y_men = np.concatenate(men_y, axis=0)

print(x_men.shape)    #(1409, 100, 100, 3)
print(y_men.shape)    #(1409,)
# exit()

all_x = []
all_y = []
for i in range(len(train)):
    ax, ay = train[i]
    all_x.append(ax)
    all_y.append(ay)

original_x = np.concatenate(all_x, axis=0)
original_y = np.concatenate(all_y, axis=0)
print(original_x.shape) #(3309, 100, 100, 3)
print(original_y.shape) #(3309,)

# exit()

augment_datagen = ImageDataGenerator(
    rescale=1/255., horizontal_flip=True, width_shift_range=0.1, rotation_range=20,
    zoom_range=1.1, fill_mode='nearest'
)

augment_size = 491
ran_idx = np.random.randint(x_men.shape[0], size=augment_size)
aug_wx = x_men[ran_idx].copy()
aug_wy = y_men[ran_idx].copy()

augment = augment_datagen.flow(
    aug_wx, aug_wy, batch_size=100, shuffle=False
)

arr_aug_x = []
arr_aug_y = []
for i in range(5):
    ax, ay = augment[i]
    arr_aug_x.append(ax)
    arr_aug_y.append(ay)
    
augment_x = np.concatenate(arr_aug_x, axis=0)
augment_y = np.concatenate(arr_aug_y, axis=0)

x = np.concatenate((original_x, augment_x))
y = np.concatenate((original_y, augment_y))

npy_path = './_data/_save_npy/keras50/men_women/'
np.save(npy_path + 'x.npy', arr=x)
np.save(npy_path + 'y.npy', arr=y)

print(x.shape, y.shape) #(3800, 100, 100, 3) (3800,)




