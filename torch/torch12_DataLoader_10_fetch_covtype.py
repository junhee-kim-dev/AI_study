
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

USE_DEVICE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_DEVICE else 'cpu')
import random
SEED = 337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

# print(x.shape)
# print(np.unique(y, return_counts=True))
# exit()


le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=SEED, stratify=y
)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

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
        # x = self.rl(x)
        x = self.l5(x)
        # x = self.sm(x)
        return x

model = Model(54, 7).to(DEVICE)


criterion = nn.CrossEntropyLoss()
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

for i in range(1000) :
    loss = fit(model, criterion, optimizer, train_loader)
    print(f"epoch : {i+1} / loss : {loss} ")

def evaluate(model, criterion, loader) :
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader :
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
    tloss = total_loss / len(loader)
    return tloss

f_loss = evaluate(model, criterion,test_loader)
# y_pred = model(x_test)
y_pred = model(x_test)
y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
y_test = y_test.cpu().numpy()
acc = accuracy_score(y_test, y_pred)

print(f_loss)
print(acc)
# 0.8460108603048114
# 0.8395480323227456