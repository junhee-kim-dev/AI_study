from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import torch
import torch.nn as nn
import torch.optim as optim

USE_DEVICE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_DEVICE else 'cpu')
path = './Study25/_data/kaggle/bank/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv')
# exit()

# train_csv[['Tenure', 'Balance']] = train_csv[['Tenure', 'Balance']].replace(0, np.nan)
# train_csv = train_csv.fillna(train_csv.mean())

# test_csv[['Tenure', 'Balance']] = test_csv[['Tenure', 'Balance']].replace(0, np.nan)
# test_csv = test_csv.fillna(test_csv.mean())
oe = LabelEncoder()       # 이렇게 정의 하는 것을 인스턴스화 한다고 함
oe.fit(train_csv['Geography'])
train_csv['Geography'] = oe.transform(train_csv['Geography'])
# test_csv['Geography'] = oe.transform(test_csv['Geography'])
oe_g = LabelEncoder()
oe_g.fit(train_csv['Gender'])
train_csv['Gender'] = oe_g.transform(train_csv['Gender'])
# test_csv['Gender'] = oe_g.transform(test_csv['Gender'])


train_csv = train_csv.drop(['CustomerId','Surname'], axis=1)
# test_csv = test_csv.drop(['CustomerId','Surname'], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

print(x.head())
print(y.head())
# exit()
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=123, stratify=y
)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
).to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y) :
    model.train()
    optimizer.zero_grad()
    hypo = model(x)
    loss = criterion(hypo, y)
    loss.backward()
    optimizer.step()
    return loss.item()

for i in range(1000) :
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f"epochs {i+1} / loss {loss}")

def evaluate(model, criterion, x, y) :
    model.eval()
    with torch.no_grad() :
        y_pred = model(x)
        f_loss = criterion(y_pred, y)
    return f_loss.item()

f_loss = evaluate(model, criterion, x_test, y_test)
result = np.round(model(x_test).detach().cpu().numpy())
y_test = y_test.detach().cpu().numpy()

acc = accuracy_score(y_test, result)
print(f"f_loss : {f_loss}")
print(f"   ACC : {acc}")
# f_loss : 0.3293099105358124
#    ACC : 0.8636694134755211