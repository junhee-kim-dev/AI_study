# tts 후 scaling 후 pca

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

datasets = load_iris()
x = datasets['data']
y = datasets.target

xb_train, xb_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=42, stratify=y
)


for i in range(4) :
    x_train = xb_train.copy()
    x_test = xb_test.copy()
    
    pca = PCA(n_components=i+1)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    model = RandomForestClassifier(random_state=333)

    model.fit(x_train, y_train)

    result = model.score(x_test, y_test)

    print('model.score:', result)

# model.score: 0.9333333333333333
# model.score: 0.9333333333333333
# model.score: 0.9333333333333333
# model.score: 0.9333333333333333
