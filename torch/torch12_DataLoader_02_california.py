from sklearn.datasets import fetch_california_housing
x, y = fetch_california_housing(return_X_y=True)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch import float32 as f32
from torch import long

import random
SEED = 337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

use = torch.cuda.is_available()
dev = torch.device('cuda' if use else 'cpu')

# print(x.shape, y.shape)
# exit()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8
)

ss = StandardScaler()
ss.fit(x_train)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

x_train = torch.tensor(x_train, dtype=f32).to(dev)
x_test = torch.tensor(x_test, dtype=f32).to(dev)
y_train = torch.tensor(y_train, dtype=f32).unsqueeze(1).to(dev)
y_test = torch.tensor(y_test, dtype=f32).unsqueeze(1).to(dev)

'''
다양한 플랫폼(특히 GPU)에서 타입 충돌을 피하기 위해,
pytorch는 target label은 무조건 int64(long)로 고정했음

회귀 : MSELoss()          -> dtype : torch.float32
이진 : BCELoss()          -> dtype : torch.float32
다중 : CrossEntropyLoss() -> dtype : torch.long
'''
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=True)

class Model(nn.Module) :    # ()안에 있는 것들은 nn.Module의 모든 것을 상속 받겠다는 뜻
    def __init__(self, input_dim, output_dim):    # 초기화 함수
            # 클래스를 쓸때는 __init__ 가 요구하는 거 다 주어져야 함
        super().__init__()               # 짧게 쓰는 방식
        # super(Model, self).__init__()  # 길게 쓰는 방식
            # Model은 클래스 이름
            # self는 super()에 있는 거 다 쓰겠다는 뜻
            # super(Model, self).__init__() 이거랑 똑같은 뜻인데 짧은거
            
        ######## 모델 정의 #########
        self.linear1 = nn.Linear(input_dim, 64)  # 정의야! 구현이 아니야!
        self.linear2 = nn.Linear(64, 32)         # 정의야! 구현이 아니야!
        self.linear3 = nn.Linear(32, 32)         # 정의야! 구현이 아니야!
        self.linear4 = nn.Linear(32, 16)         # 정의야! 구현이 아니야!
        self.linear5 = nn.Linear(16, output_dim) # 정의야! 구현이 아니야!
        self.relu = nn.ReLU()                    # 정의야! 구현이 아니야!
        self.dropout = nn.Dropout(0.2)           # 정의야! 구현이 아니야!

        ######## 모델 정의 구현 #########
    def forward(self, x) :                  
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.dropout(x)
        x = self.linear5(x)
        return x
    
model = Model(8, 1).to(dev)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader) :
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

for i in range(10) :
    verbose = train(model, criterion, optimizer, train_loader)
    print(f"epoch : {i+1} / loss : {verbose:.8f}")

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

loss = evaluate(model, criterion, test_loader)
result = model(x_test).cpu().detach().numpy()
y_test = y_test.cpu().detach().numpy()
r2 = r2_score(y_test, result)

print(f"\n최종 LOSS : {loss:.5f}")
print(f"최종   R2 : {r2:.5f}")

# 최종 LOSS : 0.27339
# 최종   R2 : 0.78542
