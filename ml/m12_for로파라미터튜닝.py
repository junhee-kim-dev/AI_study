import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier

#1. 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, train_size=0.8, stratify=y
)

learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1]
max_depth = [3, 4, 5, 6, 7]

best_score = 0

for i_1, a in enumerate(learning_rate) :
    for i_2, b in enumerate(max_depth) :
        
        param = a, b
        model = HistGradientBoostingClassifier(max_depth=b, learning_rate=a)
        model.fit(x_train, y_train)
        loss = model.score(x_test, y_test)
        
        print(f'{i_1 +1}-{i_2 +1} 회차')
        print(f'acc : {loss:.5f} [param: LR: {a} / MD: {b}]')
        
        if loss > best_score :
            best_score = loss
            best_param = f'LR: {a} / MD: {b}'
            print(f'#### 현재까지 최고: {best_param} ####')
        else :
            continue
        
print(f'최고 점수: {best_score:.5f}')
print(f'최적 변수: {best_param}')

# 1-1 회차
# acc : 0.84722 [param: LR: 0.001 / MD: 3]
# #### 현재까지 최고: LR: 0.001 / MD: 3 ####
# 1-2 회차
# acc : 0.86944 [param: LR: 0.001 / MD: 4]
# #### 현재까지 최고: LR: 0.001 / MD: 4 ####
# 1-3 회차
# acc : 0.87778 [param: LR: 0.001 / MD: 5]
# #### 현재까지 최고: LR: 0.001 / MD: 5 ####
# 1-4 회차
# acc : 0.87778 [param: LR: 0.001 / MD: 6]
# 1-5 회차
# acc : 0.87778 [param: LR: 0.001 / MD: 7]
# 2-1 회차
# acc : 0.88889 [param: LR: 0.005 / MD: 3]
# #### 현재까지 최고: LR: 0.005 / MD: 3 ####
# 2-2 회차
# acc : 0.89722 [param: LR: 0.005 / MD: 4]
# #### 현재까지 최고: LR: 0.005 / MD: 4 ####
# 2-3 회차
# acc : 0.90278 [param: LR: 0.005 / MD: 5]
# #### 현재까지 최고: LR: 0.005 / MD: 5 ####
# 2-4 회차
# acc : 0.90556 [param: LR: 0.005 / MD: 6]
# #### 현재까지 최고: LR: 0.005 / MD: 6 ####
# 2-5 회차
# acc : 0.90556 [param: LR: 0.005 / MD: 7]
# 3-1 회차
# acc : 0.89444 [param: LR: 0.01 / MD: 3]
# 3-2 회차
# acc : 0.91389 [param: LR: 0.01 / MD: 4]
# #### 현재까지 최고: LR: 0.01 / MD: 4 ####
# 3-3 회차
# acc : 0.91111 [param: LR: 0.01 / MD: 5]
# 3-4 회차
# acc : 0.91389 [param: LR: 0.01 / MD: 6]
# 3-5 회차
# acc : 0.91389 [param: LR: 0.01 / MD: 7]
# 4-1 회차
# acc : 0.93056 [param: LR: 0.05 / MD: 3]
# #### 현재까지 최고: LR: 0.05 / MD: 3 ####
# 4-2 회차
# acc : 0.95000 [param: LR: 0.05 / MD: 4]
# #### 현재까지 최고: LR: 0.05 / MD: 4 ####
# 4-3 회차
# acc : 0.95000 [param: LR: 0.05 / MD: 5]
# 4-4 회차
# acc : 0.95000 [param: LR: 0.05 / MD: 6]
# 4-5 회차
# acc : 0.94722 [param: LR: 0.05 / MD: 7]
# 5-1 회차
# acc : 0.94444 [param: LR: 0.1 / MD: 3]
# 5-2 회차
# acc : 0.95278 [param: LR: 0.1 / MD: 4]
# #### 현재까지 최고: LR: 0.1 / MD: 4 ####
# 5-3 회차
# acc : 0.96111 [param: LR: 0.1 / MD: 5]
# #### 현재까지 최고: LR: 0.1 / MD: 5 ####
# 5-4 회차
# acc : 0.95833 [param: LR: 0.1 / MD: 6]
# 5-5 회차
# acc : 0.95833 [param: LR: 0.1 / MD: 7]
# 최고 점수: 0.96111
# 최적 변수: LR: 0.1 / MD: 5
