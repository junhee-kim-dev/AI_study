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

import random
SEED = 337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

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

# print(x.head())
# print(y.head())
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

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=True)

# model = nn.Sequential(
#     nn.Linear(10, 64),
#     nn.ReLU(),
#     nn.Linear(64, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 1),
#     nn.Sigmoid()
# ).to(DEVICE)

class Model(nn.Module) :
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.l1 = nn.Linear(input_dim, 128)
        self.l2 = nn.Linear(128,64)
        self.l3 = nn.Linear(64,32)
        self.l4 = nn.Linear(32,16)
        self.l5 = nn.Linear(16, output_dim)
        self.rl = nn.ReLU()
        self.sm = nn.Sigmoid()
        self.si = nn.SiLU()

    def forward(self, x) :
        x = self.l1(x)
        x = self.rl(x)
        x = self.l2(x)
        x = self.rl(x)
        x = self.l3(x)
        x = self.rl(x)
        x = self.l4(x)
        x = self.rl(x)
        x = self.l5(x)
        x = self.sm(x)
        return x

model = Model(10, 1).to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def fit(model, criterion, optimizer, loader) :
    model.train()
    total_loss = 0
    for x, y in loader :
        optimizer.zero_grad()
        hypo = model(x)
        loss = criterion(hypo, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    tloss = total_loss / len(loader)
    return tloss

for i in range(100) :
    loss = fit(model, criterion, optimizer, train_loader)
    print(f"epochs {i+1} / loss {loss}")

def evaluate(model, criterion, loader) :
    model.eval()
    total_loss = 0
    with torch.no_grad() :
        for x, y in loader :
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
    tloss = total_loss / len(loader)
    return tloss

f_loss = evaluate(model, criterion, test_loader)
result = np.round(model(x_test).detach().cpu().numpy())
y_test = y_test.detach().cpu().numpy()

acc = accuracy_score(y_test, result)
print(f"f_loss : {f_loss}")
print(f"   ACC : {acc}")
# f_loss : 0.3293099105358124
#    ACC : 0.8636694134755211

# f_loss : 0.32616138458251953
#    ACC : 0.8639117789626757