import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

datasets = load_iris()
x = datasets['data']
y = datasets.target

pca = PCA(n_components=1)
x = pca.fit_transform(x)

# print(x.shape)  #(150, 1)
ss = StandardScaler()
x = ss.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=333)

model.fit(x_train, y_train)

result = model.score(x_test, y_test)

print(x.shape)
print('model.score:', result)


