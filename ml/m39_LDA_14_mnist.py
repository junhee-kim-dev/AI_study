from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)

x_train = x_train/255.
x_test = x_test/255.

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

for i in range(9) :
    lda = LinearDiscriminantAnalysis(n_components=i+1)
    x_train_1 = lda.fit_transform(x_train, y_train)
    x_test_1 = lda.transform(x_test)

    from xgboost import XGBClassifier
    xgb = XGBClassifier()
    xgb.fit(x_train_1, y_train)
    score = xgb.score(x_test_1, y_test)

    print(f'n_components_{i+1} :',score)    

print(np.cumsum(lda.explained_variance_ratio_))

# n_components_1 : 0.4178
# n_components_2 : 0.5654
# n_components_3 : 0.7453
# n_components_4 : 0.8309
# n_components_5 : 0.8447
# n_components_6 : 0.8683
# n_components_7 : 0.8919
# n_components_8 : 0.9094
# n_components_9 : 0.916

# [0.2392286  0.44103854 0.61953549 0.72606121 0.82012832 0.88918857 0.93892603 0.9732168  1.        ]





