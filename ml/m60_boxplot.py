import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target

df.boxplot()
plt.show()
