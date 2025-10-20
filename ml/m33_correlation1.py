import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


datasets = load_iris()

X = datasets['data']
Y = datasets.target

df = pd.DataFrame(X, columns=datasets.feature_names)

df['Target'] = Y

print(df)
print('=======짜잔=======')
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()



