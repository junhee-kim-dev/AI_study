from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import random
import numpy as np

seed = 123
random.seed(seed)
np.random.seed(seed)

x, y = load_iris(return_X_y=True)
print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, stratify=y
)

model1 = DecisionTreeClassifier(random_state=seed)
model2 = RandomForestClassifier(random_state=seed)
model3 = GradientBoostingClassifier(random_state=seed)
model4 = XGBClassifier(random_state=seed)

models = [model1, model2, model3, model4]
for model in models :
    model.fit(x_train, y_train)
    print("==========", model.__class__.__name__, "==========")
    print('acc :', model.score(x_test, y_test))
    print(model.feature_importances_)




