# 02. california
# 03. diabetes

from sklearn.datasets import fetch_california_housing, load_diabetes
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

data1 = fetch_california_housing()
data2 = load_diabetes()

datasets = [data1, data2]
dataname = ['California', 'Diabetes']

model1 = DecisionTreeRegressor(random_state=seed)
model2 = RandomForestRegressor(random_state=seed)
model3 = GradientBoostingRegressor(random_state=seed)
model4 = XGBRegressor(random_state=seed)

models = [model1, model2, model3, model4]

for i, data in enumerate(datasets):
    x = data.data
    y = data.target
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=seed,
    )
    
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    
    print(f'############ {dataname[i]} ############')

    for model in models :
        model.fit(x_train, y_train)
        print("==========", model.__class__.__name__, "==========")
        print('R2 :', model.score(x_test, y_test))
        print(np.round(model.feature_importances_,4))

# ############ California ############
# ========== DecisionTreeRegressor ==========
# R2 : 0.6047317861161444
# [0.5198 0.0425 0.047  0.0289 0.0327 0.1314 0.1009 0.0968]
# ========== RandomForestRegressor ==========
# R2 : 0.8105254207304912
# [0.5255 0.0527 0.0485 0.03   0.0329 0.1329 0.0891 0.0884]
# ========== GradientBoostingRegressor ==========
# R2 : 0.7979439328999881
# [0.6014 0.0267 0.0227 0.0051 0.0036 0.123  0.0982 0.1192]
# ========== XGBRegressor ==========
# R2 : 0.8376969113046485
# [0.4583 0.0755 0.048  0.0244 0.0279 0.1497 0.1013 0.1149]

# ############ Diabetes ############
# ========== DecisionTreeRegressor ==========
# R2 : -0.03336573740974513
# [0.0834 0.0147 0.265  0.0584 0.0442 0.0386 0.0419 0.0193 0.3734 0.0611]
# ========== RandomForestRegressor ==========
# R2 : 0.48357858805366405
# [0.0561 0.0121 0.2751 0.1006 0.0477 0.049  0.0533 0.0277 0.3063 0.0721]
# ========== GradientBoostingRegressor ==========
# R2 : 0.45674151923923667
# [0.0541 0.0133 0.2539 0.1129 0.0289 0.0422 0.0364 0.0243 0.4002 0.0337]
# ========== XGBRegressor ==========
# R2 : 0.36071718108795103
# [0.0405 0.0647 0.1651 0.0695 0.0494 0.0474 0.04   0.0905 0.3732 0.0596]


