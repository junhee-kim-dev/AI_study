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

ss = StandardScaler()
xb_train = ss.fit_transform(xb_train)
xb_test = ss.transform(xb_test)

for i in range(4) :
    x_train = xb_train.copy()
    x_test = xb_test.copy()
    
    pca = PCA(n_components=i+1)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    
    model = RandomForestClassifier(random_state=333)

    model.fit(x_train, y_train)

    result = model.score(x_test, y_test)

    print('model.score:', result)

# model.score: 0.8
# model.score: 0.8666666666666667
# model.score: 0.9333333333333333
# model.score: 0.9333333333333333

evr = pca.explained_variance_ratio_
print('evr :', evr)                 # evr : [0.7260043  0.23164585 0.03736198 0.00498787]
print('evr_sum :', sum(evr))        # evr_sum : 1.0

evr_cumsum = np.cumsum(evr)
print('evr_cumsum :',evr_cumsum)    #[0.7260043  0.95765015 0.99501213 1.        ]

import matplotlib.pyplot as plt
plt.plot(evr_cumsum)
plt.grid()
plt.show()
