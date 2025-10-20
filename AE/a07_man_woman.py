import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras import layers, models

H, W, C = 100, 100, 3

path = "./Study25/_data/image/me/"
img_pil = load_img(path + 'me.jpg', target_size=(H, W))
arr = img_to_array(img_pil).astype(float) / 255.0   # (50,50,3) in [0,1]
img = np.expand_dims(arr, axis=0)                       # (1,50,50,3)

path_load = './Study25/_data/_save_npy/keras46/gender/'
x = np.load(path_load + '(100x100)_flip_x.npy').astype(float)

x = x.reshape(-1, H, W, C)
x = x / 255.0

rng = np.random.default_rng(73)
x_noised   = np.clip(x   + rng.normal(0, 0.1, size=x.shape), 0, 1)
img_noised = np.clip(img + rng.normal(0, 0.1, size=img.shape), 0, 1)

inp = layers.Input(shape=(100, 100, 3))
x1  = layers.Conv2D(128, (2,2), activation='relu', padding='same')(inp)
x1  = layers.MaxPooling2D((2,2))(x1)
x1  = layers.Conv2D(128, (2,2), activation='relu', padding='same')(x1)
x1  = layers.MaxPooling2D((2,2))(x1)

x1  = layers.Conv2D(64, (2,2), activation='relu', padding='same')(x1)
x1  = layers.UpSampling2D((2,2))(x1)
x1  = layers.Conv2D(32, (2,2), activation='relu', padding='same')(x1)
x1  = layers.UpSampling2D((2,2))(x1)

out = layers.Conv2D(C, (2,2), activation='sigmoid', padding='same')(x1)
model = models.Model(inp, out)

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_noised, x, epochs=30, batch_size=256, validation_split=0.1, shuffle=True)

y_pred = model.predict(img_noised)

# (옵션) 시각화
plt.subplot(1,3,1); plt.title("noised");   plt.imshow(img_noised[0]); plt.axis('off')
plt.subplot(1,3,3); plt.title("clean");    plt.imshow(img[0]);        plt.axis('off')
plt.show()