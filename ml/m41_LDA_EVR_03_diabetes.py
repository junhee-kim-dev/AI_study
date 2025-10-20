
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np

from sklearn.datasets import load_diabetes
#1. 데이터
DS = load_diabetes()

x = DS.data
y = DS.target

y_ori = y.copy()
y = np.rint(y).astype(int)
# print(np.unique(y, return_counts=True))

x_trn, x_tst, y_trn, y_tst, ory_train, ory_test = train_test_split(
    x, y, y_ori,
    train_size=0.85,
    shuffle=True,
    random_state=777
)

ss = StandardScaler()
x_trn = ss.fit_transform(x_trn)
x_tst = ss.transform(x_tst)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


# pca = PCA(n_components=10)
# x_train = pca.fit_transform(x_trn)
# x_test = pca.transform(x_tst)
# pca_evr = pca.explained_variance_ratio_
# print(np.cumsum(pca_evr))
# [0.40446736 0.55664418 0.67980499 0.76976337 0.83588212 0.89678787
#  0.94905333 0.99093539 0.99918914 1.        ]

lda = LinearDiscriminantAnalysis(n_components=10)
x_train = lda.fit_transform(x_trn, y_trn)
x_test = lda.transform(x_tst)
pca_evr = lda.explained_variance_ratio_
print(np.cumsum(pca_evr))
# [0.23895161 0.35947417 0.47942599 0.58443323 0.67616117 0.76530032
#  0.83833782 0.89971643 0.95445582 1.        ]

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# model = RandomForestRegressor(random_state=333)
# model.fit(x_trn, ory_train)
# results = model.score(x_tst, ory_test)
# print(f'{model} : {results}')
# 원값 RandomForestRegressor(random_state=333) : 0.32324371338265157
# PCA  RandomForestRegressor(random_state=333) : 0.32324371338265157
# LDA  RandomForestRegressor(random_state=333) : 0.46727712026437973

model = RandomForestClassifier(random_state=333)
model.fit(x_train, y_trn)
results = model.score(x_test, y_tst)
print(f'{model} : {results}')
# PCA RandomForestClassifier(random_state=333) : 0.014925373134328358
# LDA RandomForestClassifier(random_state=333) : 0.0


