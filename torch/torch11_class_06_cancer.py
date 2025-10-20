import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import random
SEED = 337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print("torch :", torch.__version__, "사용 device :", DEVICE)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=333, stratify=y
)

scaler  = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
x_test  = torch.tensor( x_test, dtype=torch.float32).to(DEVICE)
y_test  = torch.tensor( y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# print(x_train.size(), y_train.size())
# print(x_test.size(), y_test.size())
# torch.Size([455, 30]) torch.Size([455, 1])
# torch.Size([114, 30]) torch.Size([114, 1])

# model = nn.Sequential(
#     nn.Linear(30, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.SiLU(),
#     nn.Linear(16, 1),
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
        x = self.si(x)
        x = self.l5(x)
        x = self.sm(x)
        return x

model = Model(30, 1).to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y) :
    model.train()
    optimizer.zero_grad()
    hypo = model(x)
    loss = criterion(hypo, y)
    loss.backward()
    optimizer.step()
    return loss.item()

for i in range(1000):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f"{i+1} epoch loss = {loss:.6f}")

def evaluate(model, criterion, x, y) :
    model.eval()
    with torch.no_grad() :
        y_pred = model(x)
        f_loss = criterion(y, y_pred)
    return f_loss.item()

f_loss = evaluate(model, criterion, x_test, y_test)
result = model(x_test)
y_test = y_test.detach().cpu().numpy()
result = np.round(result.detach().cpu().numpy())
acc = accuracy_score(y_test, result)

print(f_loss)
print(acc)
# 2.307284355163574
# 0.9824561403508771

# 2.801010847091675
# 0.9736842105263158
