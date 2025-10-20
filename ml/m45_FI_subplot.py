from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import random
import numpy as np

seed = 123
random.seed(seed)
np.random.seed(seed)

data1 = load_breast_cancer()
data2 = load_wine()
data3 = load_iris()
data4 = fetch_california_housing()

datasets = [data1, data2, data3, data4]
dataname = ['Cancer', 'Wine', 'Iris', 'Cali']

model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)
model5 = DecisionTreeRegressor(random_state=seed)
model6 = RandomForestRegressor(random_state=seed)
model7 = GradientBoostingRegressor(random_state=seed)
model8 = XGBRegressor(random_state=seed)

classifiers = [model1, model2, model3, model4]
regressors = [model5, model6, model7, model8]

trained_models = []
model_names = []

for i, data in enumerate(datasets):
    x = data.data
    y = data.target
    print(f'############ {dataname[i]} ############')

    if data == [data1, data2] :
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=seed, stratify=y
        )
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
        
        for model in classifiers :
            model.fit(x_train, y_train)
            trained_models.append(model)
            model_names.append(f"{dataname[i]} - {model.__class__.__name__}")
            print("==========", model.__class__.__name__, "==========")
    
    else :
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=seed
        )
        for model in regressors :
            model.fit(x_train, y_train)
            trained_models.append(model)
            model_names.append(f"{dataname[i]} - {model.__class__.__name__}")
            print("==========", model.__class__.__name__, "==========")

import matplotlib.pyplot as plt
def plot_feature_importances_grid(models, names, datasets):
    plt.figure(figsize=(14, 10))
    for idx, model in enumerate(models):
        plt.subplot(2, 2, idx + 1)
        n_features = model.feature_importances_.shape[0]
        plt.barh(range(n_features), model.feature_importances_, align='center')
        plt.xlabel("Importance")
        plt.title(names[idx])
        plt.tight_layout()
    plt.show()
    
plot_feature_importances_grid(trained_models[:4], model_names[:4], datasets)
