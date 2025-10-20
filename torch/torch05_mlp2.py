
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBRegressor

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

print("torch :", torch.__version__, "사용 DEVICE :", DEVICE)
# torch : 2.7.1+cu128 사용 DEVICE : cuda

#1. 데이터
x = np.array([[1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
              [9,  8,  7,  6,  5,  4,  3,  2,  1,  0]]).transpose()
y = np.array([1,2,3,4,5,6,7,8,9,10])

# x = torch.FloatTensor(x).to(DEVICE)
# y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)

x_scaled = (x - torch.mean(x)) / torch.std(x)

model = nn.Sequential(
    nn.Linear(3,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.Linear(3,2),
    nn.Linear(2,1)
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    hypo = model(x)
    loss = criterion(hypo, y)
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs=1000
for epoch in range(1, epochs+1) :
    loss = train(model, criterion, optimizer, x_scaled, y)
    print(f"epoch : {epochs}")
    print(f" loss : {loss:.6f}\n")

def eval(model, criterion, x, y) :
    model.eval()
    
    with torch.no_grad() :
        y_pred = model(x)
        final_loss = criterion(y, y_pred)
    
    return final_loss.item()
        
final_loss = eval(model, criterion, x_scaled, y)
print(f"최종 손실 : {final_loss:.6f}")

x_pred = torch.tensor([[10, 1.9, 0]]).to(DEVICE)
x_pred_scaled = (x_pred - torch.mean(x)) / torch.std(x)

result = model(x_pred_scaled)

print(f"[10,1.9,0]의 예측값 : {result.item():.6f}")

# 최종 손실 : 0.000000
# [10,1.9,0]의 예측값 : 10.000000