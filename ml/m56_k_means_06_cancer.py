from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd

seed = 123
random.seed(seed)
np.random.seed(seed)

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.cluster import KMeans
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

model = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=seed)    # n_clusters : 라벨 수

y_pred = model.fit_predict(x_train)

print(y_pred[:10])
print(y_train[:10])
acc = accuracy_score(y_train, y_pred)
print(acc)
exit()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('ACC :', acc)
f1 = f1_score(y_test, y_pred)
print('F1  :', f1)

# ACC : 0.9736842105263158
# F1  : 0.9793103448275862





