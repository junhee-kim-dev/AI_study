from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

(x_train, _), (x_test, _) = mnist.load_data()

# print(x_train.shape, x_test.shape) (60000, 28, 28) (10000, 28, 28)

x = np.concatenate([x_train, x_test], axis=0)
# print(x.shape)  (70000, 28, 28)

x = x.reshape(-1,28*28)
# print(x.shape)  (70000, 784)

pca = PCA(n_components=28*28)
x = pca.fit_transform(x)

evr = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(evr)
# print(evr_cumsum)

#1. 1.0일 떄 몇개?
threshold = 1.0
compo_1 = np.argmax(evr_cumsum>= threshold) +1 
print(compo_1) # 713

#2. 0.999 이상 몇개?
threshold = 0.999
compo_2 = np.argmax(evr_cumsum>= threshold) +1 
print(compo_2) # 486

#3. 0.99 이상 몇개?
threshold = 0.99
compo_3 = np.argmax(evr_cumsum>= threshold) +1 
print(compo_3) # 331

#3. 0.99 이상 몇개?
threshold = 0.95
compo_4 = np.argmax(evr_cumsum>= threshold) +1 
print(compo_4)  # 154

