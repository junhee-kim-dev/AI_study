from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
import numpy as np
import pandas as pd

data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True)]

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

models = [LinearSVC(C=0.3), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]

for i, (x, y) in enumerate(data_list, start=1):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    print('')
    print(f'#####{i}회차#####')
    
    for model in models :
        model.fit(x_train, y_train, )
        results = model.score(x_test, y_test)
        print(f'{model} :', np.round(results, 4))
  
  
'''
#####1회차#####
LinearSVC(C=0.3) : 0.9737
LogisticRegression() : 1.0
DecisionTreeClassifier() : 1.0
RandomForestClassifier() : 1.0

#####2회차#####
LinearSVC(C=0.3) : 0.979
LogisticRegression() : 0.979
DecisionTreeClassifier() : 0.9371
RandomForestClassifier() : 0.972

#####3회차#####
LinearSVC(C=0.3) : 0.9689
LogisticRegression() : 0.9711
DecisionTreeClassifier() : 0.8822
RandomForestClassifier() : 0.9733

#####4회차#####
LinearSVC(C=0.3) : 0.9778
LogisticRegression() : 0.9778
DecisionTreeClassifier() : 0.9556
RandomForestClassifier() : 1.0
'''