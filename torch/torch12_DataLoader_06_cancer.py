import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import warnings
import random

warnings.filterwarnings('ignore')

USE_DEVICE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_DEVICE else 'cpu')
SEED = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state= SEED, train_size=0.9
)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

x_train = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
x_test = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

########################## torch 데이터셋 만들기 ################################
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader         # 여기서 배치 사이즈

# 1. x와 y를 합친다.
train_set = TensorDataset(x_train, y_train)     # 합친 형태는 튜플
test_set = TensorDataset(x_test, y_test)
# print(train_set)        # <torch.utils.data.dataset.TensorDataset object at 0x713f3ab9fa30>
# print(type(train_set))  # <class 'torch.utils.data.dataset.TensorDataset'>
# print(len(train_set))   # 512
# print(train_set[0])
# (tensor([ 1.0450,  0.2971,  1.0104,  0.9031,  0.5264,  0.4881,  0.3798,  0.9817,
#          0.6636, -0.3026,  0.1428, -0.6120,  0.0699,  0.1663,  0.0701,  0.1058,
#         -0.2324,  0.4646, -0.7710, -0.1753,  0.9321, -0.0470,  0.8574,  0.7549,
#          0.7246,  0.7589,  0.2842,  1.2184,  0.2981,  0.0838], device='cuda:0'), tensor([0.], device='cuda:0'))
# print(train_set[0][0])
# tensor([ 1.3335,  0.1530,  1.1929,  1.2759, -0.4917, -0.8476, -0.1042,  0.2532,
#         -0.9334, -1.8318, -0.2748, -0.6890, -0.2432, -0.0797,  0.1937, -0.7943,
#         -0.1843,  0.4599, -0.1499, -0.7854,  0.7610, -0.2201,  0.6448,  0.6191,
#         -0.3379, -0.8659, -0.2441,  0.2271, -0.5264, -1.4602], device='cuda:0')
# print(train_set[0][1])
# tensor([0.], device='cuda:0')
# print(train_set[511]) # 된다.
# print(train_set[512]) # 길이 넘어서 없음

# 2. batch_size 를 정의한다.
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, )
test_loader = DataLoader(test_set, batch_size=100, shuffle=False, ) # train_loader 랑 batch_size 달라도 괜찮음
# print(len(train_loader))    # 6
# print(train_loader)         # <torch.utils.data.dataloader.DataLoader object at 0x706fcf12c0d0>
# print(type(train_loader))   # <class 'torch.utils.data.dataloader.DataLoader'>
# print(train_loader[0])      # 에러
# print(train_loader[0][0])   # 에러

# 3. 이터레이터 데이터 확인하기
# 1) for문으로 확인
# for aaa in train_loader :
#     print(aaa)
#     break
# 1 batch를 보여줌
# for x_batch, y_batch in train_loader :
#     print(x_batch)
#     print(y_batch)
#     break
# 1 batch를 x / y로 나눠서 줌
# exit()
# 2) next()
# bbb = iter(train_loader)
# aaa = bbb.next()    # 파이썬 버전 업 후 .next()는 삭제
# aaa = next(bbb)
# print(aaa)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 64)
        self.l4 = nn.Linear(64, 32)
        self.l5 = nn.Linear(32, output_dim)
        self.rl = nn.ReLU()
        self.sig = nn.Sigmoid()
        
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
        x = self.sig(x)
        return x

model = Model(30, 1).to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr= 0.001)


def train(model, criterion, optimizer, loader) :
    model.train()
    total_loss = 0
    
    for x_batch, y_batch in loader :
        optimizer.zero_grad()
        hypo = model(x_batch)
        loss = criterion(hypo, y_batch)
        
        loss.backward()
        optimizer.step()
        
        # total_loss = total_loss + loss.item()
        total_loss += loss.item()
    
    return total_loss / len(loader)


for i in range(1000) :
    loss = train(model, criterion, optimizer, train_loader)
    print(f"epoch: {i +1} / loss: {loss}")

def evaluate(model, criterion, loader) :
    model.eval()
    
    total_loss = 0
    for x_batch, y_batch in loader :
        with torch.no_grad():
            y_pred = model(x_batch)
            f_loss = criterion(y_pred, y_batch)
            total_loss += f_loss.item()
            
    return total_loss / len(loader)

f_loss = evaluate(model, criterion, test_loader)
results = model(x_test)

from sklearn.metrics import accuracy_score
results = np.round(results.detach().cpu().numpy())
y_test = y_test.detach().cpu().numpy()
acc = accuracy_score(y_test, results)

print(f_loss)
print(acc)
# 0.11488928645849228
# 0.9824561403508771

